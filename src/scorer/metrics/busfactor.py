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
import requests
from pathlib import Path
from collections import defaultdict, Counter
from urllib.parse import urlparse

# Optional: resolve HuggingFace → GitHub
try:
    from huggingface_hub import HfApi, get_token
    HF = HfApi()
except ImportError:
    HF = None

logger = logging.getLogger(__name__)

SINCE_DAYS_DEFAULT = 1095 # Approx. 3 years
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
    """
    APPROXIMATION: Treats the entire repo as a single unit to calculate Bus Factor
    based on commit volume, avoiding the need for file diffs/parsing.
    """
    since_dt = dt.datetime.utcnow() - dt.timedelta(days=since_days)

    dl: defaultdict[str, defaultdict[str, int]] = defaultdict(lambda: defaultdict(int)) # type: ignore
    total_by_file = defaultdict(int)
    contributors = defaultdict(set)
    file_creators: dict[str, str] = {} # Not used heavily in this approx, but kept for type safety

    if not HF:
        raise ImportError("huggingface_hub is not installed or failed to import")

    try:
        # Fetch high-level commit list (Cheap API call)
        commits = list(HF.list_repo_commits(repo_id=repo_id, repo_type=repo_type))
    except Exception as e:
        logger.error(f"Failed to list commits for {repo_id}: {e}")
        return {}, {}, {}, {}

    # Ensure timezone awareness compatibility
    since_dt_aware = since_dt.replace(tzinfo=dt.timezone.utc)
    
    # We use a dummy filename to represent the whole project
    virtual_filename = f"Entire_Repo:{repo_id}"

    # Iterate through commits (No HTTP requests inside loop!)
    for c in commits:
        # 1. Date Check
        c_date = c.created_at
        if c_date.tzinfo is None:
            c_date = c_date.replace(tzinfo=dt.timezone.utc)

        if c_date < since_dt_aware:
            continue

        # 2. Identify Author
        # Try to get author name from object, fallback to title or unknown
        # Note: HF commit objects often don't have 'author' filled if not a signed-up user
        author_names = ["unknown"]
        if hasattr(c, 'authors') and c.authors:
            author_names = c.authors
        elif hasattr(c, 'author') and c.author:
             author_names = [c.author]
        
        # 3. Populate stats for the "Virtual File"
        total_by_file[virtual_filename] += 1
        
        for author in author_names:
            auth_norm = author.lower()
            dl[virtual_filename][auth_norm] += 1
            contributors[virtual_filename].add(auth_norm)

            # Set creator as the first person found (since we iterate new->old or old->new)
            # This matters less for the approximation
            if virtual_filename not in file_creators:
                file_creators[virtual_filename] = auth_norm

    return dl, total_by_file, contributors, file_creators


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
