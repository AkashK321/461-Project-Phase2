"""
Implementing dataset quality metric scoring by looking at
the number of downloads, likes
"""

import os
import time
import logging
from dotenv import load_dotenv
from huggingface_hub import HfApi, login
from .base import get_repo_id
import math
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

load_dotenv()
# HF_TOKEN = os.getenv("HF_Token")
HF_API = HfApi()
# login(token=HF_TOKEN)

# Downloads and likes targets for top tier quality
max_downloads = 1000000  # 1 million downloads
max_likes = 2000


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


def normalize(value: int, target: int) -> float:
    if value <= 0:
        return 0.0
    return min(1.0, math.log10(value + 1) / math.log10(target + 1))


def get_dataset_quality_score(url: str, url_type: str) -> Tuple[Optional[float], int]:
    _maybe_login()
    start_time = time.time()
    logger.info(f"Starting dataset quality scoring for {url}")

    if url_type != "dataset":
        logger.warning("Dataset quality score is only applicable to datasets")
        latency = int((time.time() - start_time) * 1000)
        return None, latency

    # Get repo id
    try:
        repo_id = get_repo_id(url, url_type)
        logger.info(f"Repo ID for {url} is {repo_id}")
    except Exception as e:
        logger.error(f"Error getting repo id for {url}: {e}")
        latency = int((time.time() - start_time) * 1000)
        return None, latency

    dataset_info = HF_API.dataset_info(repo_id=repo_id, files_metadata=False)
    try:
        dataset_info = HF_API.dataset_info(repo_id=repo_id, files_metadata=False)
    except Exception as e:
        logger.error(f"Could not fetch dataset info for {repo_id}: {e}")
        latency = int((time.time() - start_time) * 1000)
        return 0.0, latency

    # Look at number of downloads and likes
    downloads = getattr(dataset_info, "downloads", 0) or 0
    likes = getattr(dataset_info, "likes", 0) or getattr(dataset_info, "stars", 0) or 0
    logger.info(f"Dataset {repo_id} has {downloads} downloads and {likes} likes.")

    # Normalize these scores
    downloads_score = normalize(downloads, max_downloads)
    likes_score = normalize(likes, max_likes)

    # If likes/stars don't exist, return downloads score
    if likes == 0:
        return round(downloads_score, 2)
    score = 0.0
    if likes == 0 and downloads > 0:
        score = downloads_score
    elif downloads > 0 or likes > 0:
        # Weighted sum of downloads and likes scores
        score = 0.8 * downloads_score + 0.2 * likes_score

    # Weighted sum of downloads and likes scores
    score = 0.8 * downloads_score + 0.2 * likes_score

    logger.info(f"Final dataset quality score for {repo_id}: {score:.2f}")
    latency = int((time.time() - start_time) * 1000)

    return round(score, 2), latency
