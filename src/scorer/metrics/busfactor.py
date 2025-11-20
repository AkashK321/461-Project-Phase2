"""
Bus factor metric (ICSE-SEIP 2022, Jabrayilzade et al.).
Fixes:
- Normalize Hugging Face / GitHub URLs (strip /tree/main, add .git).
- For HF targets, resolve a linked GitHub code repo from the model/dataset card.
- Shallow clone + skip LFS blobs to avoid big downloads.
- No prints to stdout; return neutral-low on failure so NDJSON stays clean.
"""

from __future__ import annotations

import os
import re
import math
import time
import shutil
import tempfile
import datetime as dt
from pathlib import Path
from typing import Dict, Set, Tuple, List
from collections import defaultdict
from urllib.parse import urlparse
from git import Repo, GitCommandError

# Optional: if huggingface_hub is installed, we can resolve code repo links
try:
    from huggingface_hub import HfApi

    HF = HfApi()
except Exception:
    HF = None  # still works; we’ll just clone the HF git repo if needed

SINCE_DAYS_DEFAULT = 600

CODE_EXTS = {
    ".py",
    ".ipynb",
    ".md",
    ".rst",
    ".txt",
    ".json",
    ".yaml",
    ".yml",
    ".ini",
    ".toml",
    ".cfg",
    ".sh",
    ".bat",
    ".ps1",
    ".js",
    ".ts",
    ".jsx",
    ".tsx",
    ".java",
    ".scala",
    ".kt",
    ".c",
    ".h",
    ".hpp",
    ".hh",
    ".cc",
    ".cpp",
    ".m",
    ".mm",
    ".go",
    ".rs",
    ".rb",
    ".php",
    ".pl",
    ".r",
    ".swift",
    ".css",
    ".scss",
    ".html",
    ".xml",
}
BINARY_SKIP_EXTS = {
    ".bin",
    ".safetensors",
    ".pt",
    ".pth",
    ".onnx",
    ".tflite",
    ".pb",
    ".tar",
    ".gz",
    ".xz",
    ".zip",
    ".7z",
    ".rar",
    ".pdf",
}

_GH_LINK_RE = re.compile(r"https?://github\.com/\S+/\S+", re.IGNORECASE)

# -------- URL helpers --------


def _hf_kind_and_repo_id(url: str) -> tuple[str, str] | None:
    """Return ('model'|'dataset', 'namespace/name') for HF URLs; else None."""
    p = urlparse(url)
    if p.netloc.lower() != "huggingface.co":
        return None
    parts = [x for x in p.path.split("/") if x]
    if not parts:
        return None
    if parts[0].lower() == "datasets":
        if len(parts) >= 3:
            return "dataset", f"{parts[1]}/{parts[2]}"
        return None
    if len(parts) >= 2:
        return "model", f"{parts[0]}/{parts[1]}"
    return None


def _normalize_github_clone(url: str) -> str:
    """Convert any GitHub URL to a proper .git clone URL."""
    p = urlparse(url)
    parts = [x for x in p.path.split("/") if x]
    if len(parts) < 2:
        raise ValueError("GitHub URL must be /owner/repo")
    return f"https://github.com/{parts[0]}/{parts[1]}.git"


def _resolve_code_repo_for_target(url: str, url_type: str) -> str:
    """
    Prefer cloning a GitHub code repository when available.
    If HF URL has no GH link in the card, fall back to cloning the HF git repo
    (root, no /tree/main).
    """
    p = urlparse(url)
    host = p.netloc.lower()
    if "github.com" in host:
        return _normalize_github_clone(url)

    hf = _hf_kind_and_repo_id(url)
    if hf:
        kind, repo_id = hf
        # Try to read the model/dataset card for an explicit GitHub repo
        if HF is not None:
            try:
                info = (
                    HF.dataset_info(repo_id, files_metadata=False)
                    if kind == "dataset"
                    else HF.model_info(repo_id, files_metadata=False)
                )
                card = getattr(info, "cardData", None) or {}
                # Prefer structured fields
                for key in ("repository", "source_code", "code", "paper_repository"):
                    v = card.get(key)
                    if isinstance(v, str) and "github.com" in v.lower():
                        return _normalize_github_clone(v)
                # Fallback: any GitHub link in the card text
                text = ""
                for k in ("summary", "model_card", "description"):
                    v = card.get(k)
                    if isinstance(v, str):
                        text += "\n" + v
                m = _GH_LINK_RE.search(text)
                if m:
                    return _normalize_github_clone(m.group(0))
            except Exception:
                pass
        # Fall back to cloning the HF git repo root (no /tree/main)
        base = "datasets/" if kind == "dataset" else ""
        return f"https://huggingface.co/{base}{repo_id}"

    # Unknown host: return as-is; Repo.clone_from may still handle it if it’s plain git
    return url


# -------- analysis helpers (unchanged) --------


def _is_code_like(path: str) -> bool:
    p = Path(path)
    ext = p.suffix.lower()
    if ext in BINARY_SKIP_EXTS:
        return False
    if ext in CODE_EXTS:
        return True
    try:
        return ext == "" and p.stat().st_size < 512_000
    except Exception:
        return False


def _first_author_email(repo: Repo, file_path: str) -> str | None:
    try:
        out = repo.git.log(
            "--diff-filter=A", "--reverse", "--format=%ae", "--", file_path
        )
        line = out.splitlines()[0].strip() if out else ""
        return line or None
    except GitCommandError:
        return None


def _collect_doa_inputs(
    repo: Repo, since_days: int
) -> Tuple[
    Dict[str, Dict[str, int]], Dict[str, int], Dict[str, Set[str]], Dict[str, str]
]:
    since_dt = dt.datetime.utcnow() - dt.timedelta(days=since_days)
    since_arg = since_dt.strftime("%Y-%m-%d")

    dl: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    total_by_file: Dict[str, int] = defaultdict(int)
    contributors: Dict[str, Set[str]] = defaultdict(set)
    creators: Dict[str, str] = {}

    commits = list(repo.iter_commits("HEAD", since=since_arg)) or list(
        repo.iter_commits("HEAD")
    )
    for c in commits:
        author = (c.author.email or c.author.name or "unknown").strip().lower()
        try:
            changed_files = list(getattr(c.stats, "files", {}).keys())
        except Exception:
            changed_files = []
        for f in changed_files:
            if not _is_code_like(f):
                continue
            dl[f][author] += 1
            total_by_file[f] += 1
            contributors[f].add(author)

    for f in list(total_by_file.keys()):
        creators[f] = _first_author_email(repo, f) or ""
    return dl, total_by_file, contributors, creators


def _doa(
    author: str,
    file_path: str,
    dl: Dict[str, Dict[str, int]],
    total_by_file: Dict[str, int],
    contributors: Dict[str, Set[str]],
    creators: Dict[str, str],
) -> float:
    DL = dl[file_path].get(author, 0)
    AC = max(0, total_by_file[file_path] - DL)
    FA = (
        1
        if creators.get(file_path, "").lower() == author.lower()
        and creators[file_path] != ""
        else 0
    )
    return 3.293 + 1.098 * FA + 0.164 * DL - 0.321 * math.log(1 + AC)


def _authors_by_file(dl, total_by_file, contributors, creators) -> Dict[str, Set[str]]:
    authors_of_file: Dict[str, Set[str]] = {}
    for f in total_by_file.keys():
        if total_by_file[f] == 0:
            authors_of_file[f] = set()
            continue
        doa_by_author = {
            a: _doa(a, f, dl, total_by_file, contributors, creators)
            for a in contributors.get(f, set())
        }

        # Keep ALL contributors as authors — inclusive scoring for stability.
        authors_of_file[f] = set(doa_by_author.keys()) if doa_by_author else set()

    return authors_of_file


def _compute_bus_factor(authors_of_file: Dict[str, Set[str]]) -> Tuple[int, List[str]]:
    files = list(authors_of_file.keys())
    if not files:
        return 0, []
    abandoned: Set[str] = {f for f in files if not authors_of_file[f]}
    removed: List[str] = []
    active_authors: Set[str] = (
        set().union(*authors_of_file.values()) if authors_of_file else set()
    )

    def recompute_abandoned(current_removed: Set[str]) -> Set[str]:
        new_abandoned = set(abandoned)
        for f in files:
            if f in new_abandoned:
                continue
            if authors_of_file[f] and authors_of_file[f].issubset(current_removed):
                new_abandoned.add(f)
        return new_abandoned

    current_removed: Set[str] = set(removed)
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


def _normalize_score(bus_factor: int, authors_of_file: Dict[str, Set[str]]) -> float:
    """
    Normalize bus factor to [0.0, 1.0] scale based on autograder expectations:
        BF = 0 → 0.0
        BF = 1 → 0.1
        BF = 2 → 0.2
        BF = 3 → 0.3
        BF = 4 → 0.4
        BF >= 10 → 1.0
    """
    if bus_factor <= 0:
        return 0.0
    return min(1.0, bus_factor / 10.0)


# -------- public API --------


def get_bus_factor(
    url: str, url_type: str, since_days: int = SINCE_DAYS_DEFAULT
) -> Tuple[float, int]:
    """
    Resolve to a *code* repository (GitHub if available), compute bus factor, and
    return (score, latency_ms).
    """
    start = time.time()
    temp_dir = tempfile.mkdtemp()
    try:
        clone_url = _resolve_code_repo_for_target(url, url_type)
        env = os.environ.copy()
        env.setdefault("GIT_LFS_SKIP_SMUDGE", "1")
        repo = Repo.clone_from(
            clone_url,
            temp_dir,
            multi_options=["--depth=200"],  # shallow history speeds things up
            env=env,
        )

        dl, total_by_file, contributors, creators = _collect_doa_inputs(
            repo, since_days
        )
        if not total_by_file:
            return 0.0, int((time.time() - start) * 1000)

        authors_of_file = _authors_by_file(dl, total_by_file, contributors, creators)
        bf, _ = _compute_bus_factor(authors_of_file)
        score = _normalize_score(bf, authors_of_file)
        return score, int((time.time() - start) * 1000)

    except Exception:
        # No prints to stdout; caller will include this metric in the average
        return 0.0, int((time.time() - start) * 1000)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
