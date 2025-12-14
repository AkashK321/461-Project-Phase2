"""
Implementing license metric scoring by seeing if
the license is compatible with LGPLv2.1
"""

import os
import requests
import time
import re
from dotenv import load_dotenv
from huggingface_hub import HfApi, login
from .base import get_repo_id
from typing import Tuple, Optional

# suppress logging from Hugging Face
import logging

logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

load_dotenv()
# HF_TOKEN = os.getenv("HF_Token")
HF_API = HfApi()
# login(token=HF_TOKEN)

compatible_licenses = ["apache-2.0", "mit", "bsd-2-clause", "bsd-3-clause", "lgpl-2.1"]


def _maybe_login() -> None:
    """
    Log in non-interactively only if a token is present.
    Never prompt, never run at import time.
    """
    token = (
        os.getenv("HF_TOKEN")  # preferred
        or os.getenv("HF_Token")  # be forgiving if someone used this
        or os.getenv("HUGGINGFACE_TOKEN")  # extra alias, optional
    )
    if not token:
        return
    try:
        # No interactive questions, no new session popups
        login(
            token=token,
            add_to_git_credential=False,
            write_permission=False,
            new_session=False,
        )
    except Exception:
        # Swallow login issues; callers should still work anonymously where possible
        pass


# Normalize license names from HF/GitHub API
def is_compatible(license: str) -> bool:
    if not license:
        # print("No license found")
        return False

    normalized = license.lower().strip()

    if normalized.startswith("apache"):
        normalized = "apache-2.0"
    elif normalized.startswith("mit"):
        normalized = "mit"
    elif normalized.startswith("bsd 2-clause"):
        normalized = "bsd-2-clause"
    elif normalized.startswith("bsd 3-clause"):
        normalized = "bsd-3-clause"
    elif normalized.startswith("lgpl v2.1"):
        normalized = "lgpl-2.1"

    return normalized in compatible_licenses


def get_license_score(url: str, url_type: str) -> Tuple[Optional[int], int]:
    _maybe_login()
    start_time = time.time()

    # Get repo id
    try:
        repo_id = get_repo_id(url, url_type)
    except Exception as e:
        print(f"Error getting repo id {e}")
        latency = int((time.time() - start_time) * 1000)
        return None, latency

    license = None
    if url_type == "model":
        info = HF_API.model_info(repo_id=repo_id)
        license = getattr(info, "license", None) or (info.cardData or {}).get("license")

    elif url_type == "dataset":
        info = HF_API.dataset_info(repo_id=repo_id)
        license = getattr(info, "license", None)

    elif url_type == "code":
        base_url = f"https://api.github.com/repos/{repo_id}"
        license_info = requests.get(f"{base_url}/license").json()
        license = license_info.get("license", {}).get("name")

    # print(f"License for {url} is {license}")

    # Check if it's compatible with LGPLv2.1
    normalized = is_compatible(license)

    latency = int((time.time() - start_time) * 1000)

    if normalized:
        return 1, latency
    else:
        return 0, latency


# ----------------------------
# Phase 2: /artifact/model/{id}/license-check
# ----------------------------


class LicenseCheckError(Exception):
    """I raise this when the license-check endpoint must return a non-200 status."""

    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.message = message


# accept normal GitHub repo URLs (optionally with trailing paths like /tree/main).
_GITHUB_REPO_RE = re.compile(r"^https?://(?:www\.)?github\.com/([^/]+)/([^/#?]+)")


# Minimal SPDX normalization so comparisons are stable across HF/GitHub formatting.
_SPDX_ALIASES = {
    "APACHE-2": "Apache-2.0",
    "APACHE 2.0": "Apache-2.0",
    "APACHE-2.0": "Apache-2.0",
    "MIT": "MIT",
    "BSD-3": "BSD-3-Clause",
    "BSD-3-CLAUSE": "BSD-3-Clause",
    "BSD-2": "BSD-2-Clause",
    "BSD-2-CLAUSE": "BSD-2-Clause",
    "GPL-3.0": "GPL-3.0",
    "GPL-3.0-ONLY": "GPL-3.0-only",
    "GPL-3.0-OR-LATER": "GPL-3.0-or-later",
    "LGPL-3.0": "LGPL-3.0",
    "LGPL-3.0-ONLY": "LGPL-3.0-only",
    "LGPL-3.0-OR-LATER": "LGPL-3.0-or-later",
}


def _normalize_spdx(raw: str | None) -> str | None:
    """I return a canonical SPDX-ish string, or None when the license is unknown."""
    if not raw:
        return None

    s = str(raw).strip()
    if not s:
        return None

    upper = s.upper()

    # GitHub sometimes returns these for edge cases.
    if upper in {"NOASSERTION", "OTHER", "NONE"}:
        return None

    if upper in _SPDX_ALIASES:
        return _SPDX_ALIASES[upper]

    # Common HF style: "apache-2.0", "mit", "gpl-3.0".
    if re.fullmatch(r"[a-z0-9\.-]+", s):
        return _SPDX_ALIASES.get(upper, s)

    return s


def _parse_github_repo(github_url: str) -> tuple[str, str]:
    """I extract (owner, repo) from a GitHub repository URL."""
    m = _GITHUB_REPO_RE.match(github_url.strip())
    if not m:
        raise LicenseCheckError(400, "Malformed github_url")

    owner, repo = m.group(1), m.group(2)
    repo = repo.removesuffix(".git")
    return owner, repo


def _get_github_repo_spdx(github_url: str) -> str | None:
    """I fetch the SPDX id for a GitHub repository using the GitHub REST API."""
    owner, repo = _parse_github_repo(github_url)

    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "ece461-registry",
    }

    token = os.getenv("GITHUB_TOKEN") or os.getenv("GITHUB_PAT")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    url = f"https://api.github.com/repos/{owner}/{repo}/license"

    try:
        resp = requests.get(url, headers=headers, timeout=10)
    except Exception as e:
        raise LicenseCheckError(502, f"GitHub license lookup failed: {e}")

    if resp.status_code == 404:
        raise LicenseCheckError(404, "GitHub project not found")

    if resp.status_code >= 500:
        raise LicenseCheckError(502, "GitHub license lookup failed")

    if resp.status_code != 200:
        # Includes 403 (rate limit) and 401 (bad token)
        raise LicenseCheckError(502, "GitHub license lookup failed")

    try:
        payload = resp.json()
        spdx = (payload.get("license") or {}).get("spdx_id")
        return _normalize_spdx(spdx)
    except Exception as e:
        raise LicenseCheckError(502, f"GitHub license parse failed: {e}")


def _get_hf_model_spdx(model_url: str) -> str | None:
    """I fetch the license string from HuggingFace model metadata."""
    repo_id = get_repo_id(model_url, "model")
    if not repo_id:
        raise LicenseCheckError(404, "Model not found on HuggingFace")

    try:
        info = HF_API.model_info(repo_id=repo_id)
    except Exception as e:
        raise LicenseCheckError(404, f"Model not found on HuggingFace: {e}")

    license_field = getattr(info, "license", None)
    return _normalize_spdx(license_field)


def license_check_bool(model_url: str, github_url: str) -> bool:
    """
    return True iff the GitHub repo license is compatible with the model license.

    - If the HF model has *no* license (e.g., audience-classifier), return 200 + false.
    - If GitHub project doesnt exist: 404 (raised as LicenseCheckError).
    - If GitHub/HF lookup fails: 502 (raised as LicenseCheckError).
    - Otherwise: currently treat “compatible” as “same normalized SPDX id”.
    """
    model_spdx = _get_hf_model_spdx(model_url)
    if model_spdx is None:
        return False

    gh_spdx = _get_github_repo_spdx(github_url)
    if gh_spdx is None:
        return False

    return model_spdx == gh_spdx
