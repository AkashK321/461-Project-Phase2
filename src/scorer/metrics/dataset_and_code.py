"""
Dataset and code metrics.
- Look for dataset mentions in README (documentation).
- Look for example code/scripts (training, evaluation, requirements).
"""

import os
import time
import re
from dotenv import load_dotenv
from huggingface_hub import HfApi, login
from .base import get_repo_id

load_dotenv()
# HF_TOKEN = os.getenv("HF_Token")
HF_API = HfApi()
# login(token=HF_TOKEN)


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


def get_dataset_and_code_score(url: str, url_type: str):
    _maybe_login()
    start_time = time.time()

    try:
        repo_id = get_repo_id(url, url_type)
    except Exception as e:
        print(f"Error getting repo id {e}")
        latency = int((time.time() - start_time) * 1000)
        return 0.0, latency

    # Fetch README
    try:
        if url_type == "model":
            repo_info = HF_API.model_info(repo_id=repo_id, files_metadata=True)
        elif url_type == "dataset":
            repo_info = HF_API.dataset_info(repo_id=repo_id, files_metadata=True)
        else:
            latency = int((time.time() - start_time) * 1000)
            print("dataset_and_code_score only applicable to model/dataset")
            return 0.0, latency
    except Exception as e:
        print(f"Error fetching repo info {e}")
        latency = int((time.time() - start_time) * 1000)
        return 0.0, latency

    # Extract README content (if available)
    readme = getattr(repo_info, "cardData", None) or {}
    readme_text = ""
    if readme and "README.md" in repo_info.siblings:
        if "datasets" in readme:
            readme_text += " ".join(readme.get("datasets", []))
        if "model-index" in readme:
            readme_text += " ".join(
                [m.get("name", "") for m in readme.get("model-index", [])]
            )
    else:
        readme_text = ""

    # Check for dataset mentions in README
    dataset_documented = False
    dataset_patterns = [r"dataset", r"corpus", r"benchmark", r"train set", r"eval set"]
    for pattern in dataset_patterns:
        if re.search(pattern, readme_text, re.IGNORECASE):
            dataset_documented = True
            break

    # Check for code scripts or requirements
    files = [f.rfilename for f in repo_info.siblings]
    has_code = any(
        f.endswith((".py", ".ipynb"))
        or "requirements" in f.lower()
        or "train" in f.lower()
        for f in files
    )

    score = 0.0
    if dataset_documented:
        score += 0.5
    if has_code:
        score += 0.5

    latency = int((time.time() - start_time) * 1000)
    return round(score, 2), latency
