# --- size.py (refined) -------------------------------------------------------
import os
import re
import time
from typing import Tuple, Dict, Optional, List, DefaultDict
from collections import defaultdict

import requests
from huggingface_hub import HfApi, login
from dotenv import load_dotenv

from .base import get_repo_id

import logging as _logging

_logging.getLogger("huggingface_hub").setLevel(_logging.ERROR)

load_dotenv()
HF_API = HfApi()

HARDWARE_LIMITS = {
    "raspberry_pi": 4_000_000_000,  # 4 GB
    "jetson_nano": 4_000_000_000,  # 4 GB
    "desktop_pc": 32_000_000_000,  # 32 GB
    "aws_server": 512_000_000_000,  # 512 GB
}

# Tunable: fraction of RAM we assume can hold weights
USABLE_FRACTION = {
    "raspberry_pi": 0.26,  # ~1.04 GB effective
    "jetson_nano": 0.35,  # ~1.40 GB effective
    "desktop_pc": 0.85,  # ~27.2 GB effective
    "aws_server": 0.98,  # ~501 GB effective
}

_WEIGHT_EXTS = {".safetensors", ".bin", ".pt", ".pth", ".onnx", ".tflite", ".pb"}

# Shard pattern: name-00001-of-00005.safetensors
_SHARD_RE = re.compile(
    r"^(?P<prefix>.+?)-\d{5}-of-\d{5}\.(?P<ext>[^.]+)$", re.IGNORECASE
)


def _looks_like_weight_file(name: str) -> bool:
    lower = name.lower()
    if any(
        token in lower
        for token in ("optimizer", "optim", "training", "trainer", "adam")
    ):
        return False
    return any(lower.endswith(ext) for ext in _WEIGHT_EXTS)


def _family_key(filename: str) -> str:
    """
    Group shards into families; non-sharded files use full basename as family key.
    Families also implicitly separate frameworks by extension.
    """
    m = _SHARD_RE.match(filename)
    if m:
        ext = m.group("ext").lower()
        return f"{m.group('prefix').lower()}.{ext}"
    # non-sharded files: drop directory, keep basename + ext
    base = filename.lower().split("/")[-1]
    return base


def _framework_weight(filename: str) -> str:
    fn = filename.lower()
    if fn.endswith(".safetensors"):
        return "safetensors"
    if fn.endswith(".bin") or fn.endswith(".pt") or fn.endswith(".pth"):
        return "pytorch"
    if fn.endswith(".onnx"):
        return "onnx"
    if fn.endswith(".tflite"):
        return "tflite"
    if fn.endswith(".pb"):
        return "tensorflow"
    return "other"


# Preference order if two families have the *same* total size
_FRAMEWORK_PREF = {
    "safetensors": 0,
    "pytorch": 1,
    "onnx": 2,
    "tflite": 3,
    "tensorflow": 4,
    "other": 5,
}


def _pick_min_viable_family(files: List[tuple]) -> int:
    """
    files: list of (rfilename, size)
    Return total bytes for the **smallest viable** weight family.
    """
    # 1) keep only weight-like files
    weights = [(f, int(sz or 0)) for f, sz in files if _looks_like_weight_file(f)]
    if not weights:
        # fallback: sum all files (avoid returning 0)
        return sum(int(sz or 0) for _, sz in files)

    # 2) group into families (each shard set or single file)
    families: DefaultDict[str, int] = defaultdict(int)
    fam_framework: Dict[str, str] = {}
    for fname, sz in weights:
        fkey = _family_key(fname)
        families[fkey] += sz
        fam_framework[fkey] = _framework_weight(fname)

    # 3) choose the minimal total; tie-break by preferred framework
    best_key = None
    best_total = None
    for k, total in families.items():
        if best_total is None or total < best_total:
            best_key, best_total = k, total
        elif total == best_total:
            # tie-break on framework preference
            fw_best = _FRAMEWORK_PREF.get(fam_framework.get(best_key, "other"), 99)
            fw_new = _FRAMEWORK_PREF.get(fam_framework.get(k, "other"), 99)
            if fw_new < fw_best:
                best_key, best_total = k, total
    return int(best_total or 0)


def _maybe_login() -> None:
    token = (
        os.getenv("HF_TOKEN") or os.getenv("HF_Token") or os.getenv("HUGGINGFACE_TOKEN")
    )
    if not token:
        return
    try:
        login(
            token=token,
            add_to_git_credential=False,
            write_permission=False,
            new_session=False,
        )
    except Exception:
        pass


def _hf_total_weight_bytes_model(repo_id: str) -> int:
    info = HF_API.model_info(repo_id=repo_id, files_metadata=True)
    files = [(s.rfilename or "", int(s.size or 0)) for s in info.siblings]
    return _pick_min_viable_family(files)


def _hf_total_weight_bytes_dataset(repo_id: str) -> int:
    info = HF_API.dataset_info(repo_id=repo_id, files_metadata=True)
    return sum(int(s.size or 0) for s in info.siblings)


def _github_repo_bytes(repo_id: str) -> int:
    base_url = f"https://api.github.com/repos/{repo_id}"
    code_info = requests.get(base_url).json()
    return int(code_info.get("size", 0) or 0) * 1024


# ---- Scoring curve (tuned to your expected outputs) -------------------------
_K_UNDER = 0.6  # controls drop rate while u in [0,1]


def _score_utilization(u: float) -> float:
    if u <= 0:
        return 1.0
    if u <= 1.0:
        return max(0.0, 1.0 - _K_UNDER * u)
    # Over budget: penalize quickly
    return max(0.0, 1.0 - _K_UNDER - 3.0 * (u - 1.0))


def _score_on_hardware(total_bytes: int, hw_key: str) -> float:
    limit = HARDWARE_LIMITS[hw_key]
    effective = max(1, int(limit * USABLE_FRACTION[hw_key]))
    u = total_bytes / effective
    return max(0.0, min(1.0, round(_score_utilization(u), 2)))


def get_size_score(url: str, url_type: str) -> Tuple[Optional[Dict[str, float]], int]:
    _maybe_login()
    t0 = time.time()
    try:
        repo_id = get_repo_id(url, url_type)
    except Exception as e:
        print(f"Error getting repo id {e}")
        return None, int((time.time() - t0) * 1000)

    try:
        if url_type == "model":
            total_bytes = _hf_total_weight_bytes_model(repo_id)
        elif url_type == "dataset":
            total_bytes = _hf_total_weight_bytes_dataset(repo_id)
        elif url_type == "code":
            total_bytes = _github_repo_bytes(repo_id)
        else:
            total_bytes = 0
    except Exception as e:
        print(f"Error fetching artifact size: {e}")
        total_bytes = 0

    scores = {hw: _score_on_hardware(total_bytes, hw) for hw in HARDWARE_LIMITS}
    latency = int((time.time() - t0) * 1000)
    return scores, latency


# --- end ---------------------------------------------------------------------
