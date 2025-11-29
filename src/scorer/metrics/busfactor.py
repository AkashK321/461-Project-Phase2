"""
Bus factor metric (ICSE-SEIP 2022, Jabrayilzade et al.).
"""

from __future__ import annotations

import logging
import os
import re
import math
import time
import datetime as dt
from pathlib import Path
from collections import defaultdict, Counter
from urllib.parse import urlparse

# Optional: resolve HuggingFace → GitHub
try:
    from huggingface_hub import HfApi
    HF = HfApi()
except ImportError:
    HF = None

logger = logging.getLogger(__name__)

SINCE_DAYS_DEFAULT = 600
DOA_THRESHOLD = 3.293

CODE_EXTS = {
    ".py", ".ipynb", ".md", ".rst", ".txt", ".json", ".yaml", ".yml",
    ".ini", ".toml", ".cfg", ".sh", ".bat", ".ps1", ".js", ".ts",
    ".jsx", ".tsx", ".java", ".scala", ".kt", ".c", ".h", ".hpp",
    ".hh", ".cc", ".cpp", ".m", ".mm", ".go", ".rs", ".rb", ".php",
    ".pl", ".r", ".swift", ".css", ".scss", ".html", ".xml",
}

BINARY_SKIP_EXTS = {
    ".bin", ".safetensors", ".pt", ".pth", ".onnx", ".tflite", ".pb",
    ".tar", ".gz", ".xz", ".zip", ".7z", ".rar", ".pdf",
}

_GH_LINK_RE = re.compile(r"https?://github\.com/\S+/\S+", re.IGNORECASE)

# ---------- URL helpers ----------


def _hf_kind_and_repo_id(url: str):
    p = urlparse(url)

    if p.netloc.lower() != "huggingface.co":
        return None

    parts = [x for x in p.path.split("/") if x]

    if not parts:
        return None

    # Datasets must be: /datasets/<namespace>/<repo>
    if parts[0].lower() == "datasets":
        if len(parts) >= 3:
            return "dataset", f"{parts[1]}/{parts[2]}"
        return None

    # Models must be: /<namespace>/<repo>
    if len(parts) >= 2:
        return "model", f"{parts[0]}/{parts[1]}"

    return None


def _to_hf_repo_id(url: str) -> tuple[str, str] | None:
    p = urlparse(url)
    parts = [x for x in p.path.split("/") if x]

    if "github.com" in p.netloc.lower() and len(parts) >= 2:
        return "code", f"{parts[0]}/{parts[1]}"

    return _hf_kind_and_repo_id(url)


def _resolve_code_repo_for_target(url: str, url_type: str) -> str:
    p = urlparse(url)

    if "github.com" in p.netloc.lower():
        return url  # Keep it as a GitHub URL for now

    hf = _hf_kind_and_repo_id(url)

    if hf:
        kind, repo_id = hf

        if HF is not None:
            try:
                info = (
                    HF.dataset_info(repo_id, files_metadata=False)
                    if kind == "dataset"
                    else HF.model_info(repo_id, files_metadata=False)
                )

                card = getattr(info, "cardData", None) or {}

                for key in ("repository", "source_code", "code"):
                    v = card.get(key)
                    if isinstance(v, str) and "github.com" in v.lower():
                        return v

                for field in ("summary", "description"):
                    text = card.get(field, "")
                    if isinstance(text, str):
                        match = _GH_LINK_RE.search(text)  # type: ignore
                        if match:
                            return _normalize_github_clone(match.group(0))

            except Exception:
                pass

        base = "datasets/" if kind == "dataset" else ""
        return f"https://huggingface.co/{base}{repo_id}"

    return url


def _normalize_github_clone(url: str) -> str:
    p = urlparse(url)
    parts = [x for x in p.path.split("/") if x]

    if len(parts) < 2:
        raise ValueError("GitHub URL must be /owner/repo")
    return f"https://github.com/{parts[0]}/{parts[1]}.git"


# ---------- Analysis helpers ----------


def _is_code_like(path: str) -> bool:
    p = Path(path)
    ext = p.suffix.lower()
    if ext in BINARY_SKIP_EXTS:
        return False
    if ext in CODE_EXTS:
        return True
    # If using API without local stats, we cannot rely on st_size
    # We will assume no extension = code for now, or skip if unsure.
    return ext == ""


def _collect_doa_inputs_from_hf(repo_id: str, repo_type: str, since_days: int):
    """Collects Degree-of-Authorship data from Hugging Face Hub API."""
    since_dt = dt.datetime.utcnow() - dt.timedelta(days=since_days)

    dl: defaultdict[str, defaultdict[str, int]] = defaultdict(lambda: defaultdict(int))
    total_by_file = defaultdict(int)
    contributors = defaultdict(set)
    file_creators: dict[str, str] = {}

    if not HF:
        raise ImportError("huggingface_hub is not installed or failed to import")

    try:
        # FIX 1: Use list_repo_commits (returns GitCommitInfo objects)
        commits = HF.list_repo_commits(repo_id=repo_id, repo_type=repo_type)
        logger.info(f"commits: {commits}")
    except Exception as e:
        logger.error(f"Failed to list commits for {repo_id}: {e}")
        return {}, {}, {}, {}

    # FIX 2: Correct attribute is 'created_at', not 'committed_date'
    # Ensure timezone awareness compatibility (compare both as UTC or unaware)
    recent_commits = []
    for c in commits:
        c_date = c.created_at
        if c_date.tzinfo is None:
            c_date = c_date.replace(tzinfo=dt.timezone.utc)
        if since_dt.tzinfo is None:
            since_dt = since_dt.replace(tzinfo=dt.timezone.utc)
            
        if c_date > since_dt:
            recent_commits.append(c)

    # Sort oldest to newest
    recent_commits.sort(key=lambda c: c.created_at)

    # Prepare URL prefix for API calls
    # repo_type input is "model", "dataset", "space". URL expects "models", "datasets", "spaces"
    api_type = f"{repo_type}s" if repo_type else "models"

    for commit in recent_commits:
        try:
            # FIX 3: Fetch detailed commit info (including files) via raw API
            # HfApi() does not have a helper to get the "diff" directly, 
            # so we query the API endpoint standard for commit details.
            commit_url = f"https://huggingface.co/api/{api_type}/{repo_id}/commit/{commit.commit_id}"
            
            # Use the internal session from HfApi to handle auth headers if logged in
            resp = HF.session.get(commit_url)
            if resp.status_code != 200:
                continue
                
            commit_data = resp.json()
            
            # Author email logic
            # The list_repo_commits object has 'authors', usually a list of names.
            # The raw API JSON usually has 'author' object with 'email' if available.
            author_email = "unknown"
            if "author" in commit_data and commit_data["author"]:
                 author_email = commit_data["author"].get("email") or "unknown"
            # Fallback to name if email missing
            elif len(commit.authors) > 0:
                author_email = commit.authors[0]
            
            author_email = author_email.lower()

            # FIX 4: Use the 'files' list from the API response instead of parsing diff text
            # commit_data['files'] is usually a list of dicts with 'path'
            files_changed = [f.get("path") for f in commit_data.get("files", [])]

            for f in files_changed:
                # if not f or not _is_code_like(f):
                #     continue

                dl[f][author_email] += 1
                total_by_file[f] += 1
                contributors[f].add(author_email)

                # Record the author of the first commit touching a file
                if f not in file_creators:
                    file_creators[f] = author_email

        except Exception as e:
            logger.warning(f"Skipping commit {commit.commit_id} due to error: {e}")

    creators = file_creators
    return dl, total_by_file, contributors, creators


def _doa(author, file_path, dl, total_by_file, contributors, creators):
    DL = dl[file_path].get(author, 0)
    AC = max(0, total_by_file[file_path] - DL)
    # Safely handle if creator is missing (though logic above should prevent it)
    creator = creators.get(file_path, "")
    FA = 1 if creator == author else 0

    return 3.293 + 1.098 * FA + 0.164 * DL - 0.321 * math.log(1 + AC)


def _authors_by_file(dl, total_by_file, contributors, creators):
    authors_of_file = {}

    for f in total_by_file:
        doa_scores = {
            a: _doa(a, f, dl, total_by_file, contributors, creators)
            for a in contributors.get(f, set())
        }

        authors_of_file[f] = {
            a for a, doa in doa_scores.items() if doa >= DOA_THRESHOLD
        }

    return authors_of_file


def _compute_bus_factor(authors_of_file):
    files = list(authors_of_file.keys())
    abandoned: set[str] = {f for f in files if not authors_of_file[f]}
    removed = []
    active_authors = set().union(*authors_of_file.values())

    def recompute_abandoned(current_removed):
        new_abandoned = set(abandoned)
        for f in files:
            if authors_of_file[f] and authors_of_file[f].issubset(current_removed):
                new_abandoned.add(f)
        return new_abandoned

    current_removed = set()

    while True:
        if len(abandoned) > 0.5 * len(files):
            return len(removed), removed
        if not active_authors:
            return len(removed), removed

        coverage = {
            a: sum(1 for f in files if a in authors_of_file[f]) for a in active_authors
        }

        top_author = max(coverage.items(), key=lambda kv: kv[1])[0]
        removed.append(top_author)
        current_removed.add(top_author)
        active_authors.remove(top_author)
        abandoned = recompute_abandoned(current_removed)


def _normalize_score(bus_factor: int, authors_of_file) -> float:
    if bus_factor <= 0 or not authors_of_file:
        return 0.0

    active_authors = set().union(*authors_of_file.values())

    if not active_authors:
        return 0.0

    total_authors = len(active_authors)

    score = 1.0 - (bus_factor / total_authors)

    return max(0.0, min(1.0, score))


# ---------- Public API ----------


def get_bus_factor(url: str, url_type: str, since_days: int = SINCE_DAYS_DEFAULT):
    start = time.time()
    logger.info(f"Calculating bus factor for URL: {url} (type: {url_type})")

    try:
        repo_info = _to_hf_repo_id(url)
        if not repo_info:
            logger.warning(f"Could not resolve '{url}' to a Hugging Face repo.")
            return 0.0, int((time.time() - start) * 1000)

        repo_type, repo_id = repo_info

        dl, total_by_file, contributors, creators = _collect_doa_inputs_from_hf(
            repo_id, repo_type, since_days
        )

        if not total_by_file:
            logger.info(f"No code-like files with commit history found for {repo_id}.")
            return 0.0, int((time.time() - start) * 1000)

        authors_of_file = _authors_by_file(dl, total_by_file, contributors, creators)

        bf, _ = _compute_bus_factor(authors_of_file)
        score = _normalize_score(bf, authors_of_file)
        logger.info(f"Bus factor score for {repo_id}: {score} (bus factor: {bf})")
        return score, int((time.time() - start) * 1000)

    except Exception as e:
        logger.exception(f"Error calculating bus factor for URL: {url} with Exception {e}")
        return 0.0, int((time.time() - start) * 1000)
