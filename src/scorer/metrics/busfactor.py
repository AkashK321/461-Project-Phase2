"""
Bus factor metric (ICSE-SEIP 2022, Jabrayilzade et al.).
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

# Optional: resolve HuggingFace → GitHub
try:
    from huggingface_hub import HfApi

    HF = HfApi()
except Exception:
    HF = None

SINCE_DAYS_DEFAULT = 600
DOA_THRESHOLD = 3.293

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

# ---------- URL helpers ----------


def _hf_kind_and_repo_id(url: str):
    p = urlparse(url)
    if p.netloc.lower() != "huggingface.co":
        return None
    parts = [x for x in p.path.split("/") if x]
    if not parts:
        return None
    if parts[0] == "datasets" and len(parts) >= 3:
        return "dataset", f"{parts[1]}/{parts[2]}"
    if len(parts) >= 2:
        return "model", f"{parts[0]}/{parts[1]}"
    return None


def _normalize_github_clone(url: str) -> str:
    p = urlparse(url)
    parts = [x for x in p.path.split("/") if x]
    return f"https://github.com/{parts[0]}/{parts[1]}.git"


def _resolve_code_repo_for_target(url: str, url_type: str) -> str:
    p = urlparse(url)
    if "github.com" in p.netloc.lower():
        return _normalize_github_clone(url)

    hf = _hf_kind_and_repo_id(url)
    if hf:
        kind, repo_id = hf
        if HF:
            try:
                info = (
                    HF.dataset_info(repo_id)
                    if kind == "dataset"
                    else HF.model_info(repo_id)
                )
                card = getattr(info, "cardData", None) or {}

                for key in ("repository", "source_code", "code"):
                    if key in card and "github.com" in card[key].lower():
                        return _normalize_github_clone(card[key])

                for field in ("summary", "description"):
                    text = card.get(field, "")
                    if isinstance(text, str):
                        match = _GH_LINK_RE.search(text)
                        if match:
                            return _normalize_github_clone(match.group(0))
            except Exception:
                pass

        base = "datasets/" if kind == "dataset" else ""
        return f"https://huggingface.co/{base}{repo_id}"

    return url


# ---------- Analysis helpers ----------


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


def _first_author_email(repo: Repo, file_path: str):
    try:
        out = repo.git.log(
            "--diff-filter=A", "--reverse", "--format=%ae", "--", file_path
        )
        return out.splitlines()[0].strip() if out else ""
    except GitCommandError:
        return ""


def _collect_doa_inputs(repo: Repo, since_days: int):
    since_dt = dt.datetime.utcnow() - dt.timedelta(days=since_days)
    since_arg = since_dt.strftime("%Y-%m-%d")

    dl = defaultdict(lambda: defaultdict(int))
    total_by_file = defaultdict(int)
    contributors = defaultdict(set)
    creators = {}

    commits = list(repo.iter_commits("HEAD", since=since_arg))
    if not commits:
        commits = list(repo.iter_commits("HEAD"))

    commits.sort(key=lambda c: c.committed_datetime)

    for c in commits:
        if len(c.parents) > 1:  # skip merge commits
            continue

        author = (c.author.email or "unknown").lower()
        files = c.stats.files.keys() if c.stats else []

        for f in files:
            if not _is_code_like(f):
                continue
            dl[f][author] += 1
            total_by_file[f] += 1
            contributors[f].add(author)

    for f in total_by_file.keys():
        creators[f] = _first_author_email(repo, f)

    return dl, total_by_file, contributors, creators


def _doa(author, file_path, dl, total_by_file, contributors, creators):
    DL = dl[file_path].get(author, 0)
    AC = max(0, total_by_file[file_path] - DL)
    FA = 1 if creators[file_path] == author else 0
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
    abandoned = {f for f in files if not authors_of_file[f]}
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


def _normalize_score(bf: int) -> float:
    return min(bf / 10.0, 1.0)


# ---------- Public API ----------


def get_bus_factor(url: str, url_type: str, since_days: int = SINCE_DAYS_DEFAULT):
    start = time.time()
    temp_dir = tempfile.mkdtemp()

    try:
        clone_url = _resolve_code_repo_for_target(url, url_type)

        env = os.environ.copy()
        env["GIT_LFS_SKIP_SMUDGE"] = "1"

        repo = Repo.clone_from(
            clone_url,
            temp_dir,
            multi_options=["--filter=blob:none"],
            env=env,
        )

        dl, total_by_file, contributors, creators = _collect_doa_inputs(
            repo, since_days
        )

        if not total_by_file:
            return 0.0, int((time.time() - start) * 1000)

        authors_of_file = _authors_by_file(dl, total_by_file, contributors, creators)
        bf, _ = _compute_bus_factor(authors_of_file)
        score = _normalize_score(bf)

        return score, int((time.time() - start) * 1000)

    except Exception:
        return 0.0, int((time.time() - start) * 1000)

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
