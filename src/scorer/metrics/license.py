"""
Implementing license metric scoring by seeing if
the license is compatible with LGPLv2.1
"""

import os
import requests
import time
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
