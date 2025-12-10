"""
Dataset and code metrics.
- Look for dataset mentions in README (documentation).
- Look for example code/scripts (training, evaluation, requirements).
"""

import logging
import os
import time
import re
import logging
from dotenv import load_dotenv
from huggingface_hub import HfApi, login, hf_hub_download
from .base import get_repo_id

logger = logging.getLogger(__name__)

load_dotenv()
HF_API = HfApi()
# login(token=HF_TOKEN)
logger = logging.getLogger(__name__)


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


def get_dataset_and_code_score(url: str, url_type: str):
    _maybe_login()
    start_time = time.time()

    logger.info(f"Calculating dataset_and_code score for {url} of type {url_type}")
    try:
        repo_id = get_repo_id(url, url_type)
    except Exception as e:
        logger.warning(f"Error getting repo id {e}")
        latency = int((time.time() - start_time) * 1000)
        return 0.0, latency

    # Fetch Info
    try:
        if url_type == "model":
            repo_info = HF_API.model_info(repo_id=repo_id, files_metadata=True)
        elif url_type == "dataset":
            repo_info = HF_API.dataset_info(repo_id=repo_id, files_metadata=False)
        else:
            latency = int((time.time() - start_time) * 1000)
            logger.info("dataset_and_code_score only applicable to model/dataset")
            return 0.0, latency
    except Exception as e:
        logger.warning(f"Error fetching repo info {e}")
        latency = int((time.time() - start_time) * 1000)
        return 0.0, latency

    # Extract README content (if available)
    readme_text = ""
    try:
        # Check if README.md exists in the file list first
        if "README.md" in [f.rfilename for f in repo_info.siblings]:
            readme_path = hf_hub_download(
                repo_id=repo_id,
                filename="README.md",
                repo_type=url_type # "model" or "dataset"
            )
            with open(readme_path, "r", encoding="utf-8") as f:
                readme_text = f.read()
    except Exception as e:
        logger.warning(f"Error downloading README: {e}")
        # Proceed with empty text or handle error

    # Check for dataset mentions in README
    dataset_documented = False
    dataset_patterns = [r"dataset", r"corpus", r"benchmark", r"train set", r"eval set"]
    for pattern in dataset_patterns:
        if re.search(pattern, readme_text, re.IGNORECASE):
            dataset_documented = True
            break
    logger.info(f"Dataset documented: {dataset_documented}")

    # Check for code scripts or requirements
    local_code_found = False
    # This regex looks for standard github.com/user/repo patterns
    github_link_found = False
    github_pattern = r"https?://(?:www\.)?github\.com/[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+"
    if re.search(github_pattern, readme_text):
        github_link_found = True
        logger.info("GitHub link found in README")

    if url_type == "model":
        # We have siblings from the API call
        files = [f.rfilename for f in repo_info.siblings]
        local_code_found = any(
            f.endswith((".py", ".ipynb"))
            or "requirements" in f.lower()
            or "train" in f.lower()
            for f in files
        )
    elif url_type == "dataset":
        common_code_files = [
            "requirements.txt",
            "setup.py",
            "train.py",
            "scripts/train.py",
            "eval.py",
        ]

        for filename in common_code_files:
            try:
                # file_exists is efficient (HEAD request)
                if HF_API.file_exists(
                    repo_id=repo_id, filename=filename, repo_type="dataset"
                ):
                    local_code_found = True
                    break
            except Exception:
                continue

    has_code = local_code_found or github_link_found

    score = 0.0
    if dataset_documented:
        score += 0.5
    if has_code:
        score += 0.5

    latency = int((time.time() - start_time) * 1000)
    return round(score, 2), latency
