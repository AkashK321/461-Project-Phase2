"""
Dataset and code metrics.
- Look for dataset mentions in README (documentation).
- Look for example code/scripts (training, evaluation, requirements).
"""

import os
import time
import re
import logging
from dotenv import load_dotenv
from huggingface_hub import HfApi, login
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
from .base import get_repo_id

logger = logging.getLogger(__name__)

load_dotenv()
HF_API = HfApi()

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

    try:
        repo_id = get_repo_id(url, url_type)
    except Exception as e:
        print(f"Error getting repo id {e}")
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
            return 0.0, latency
    except (GatedRepoError, RepositoryNotFoundError):
        latency = int((time.time() - start_time) * 1000)
        return 0.0, latency
    except Exception as e:
        print(f"Error fetching repo info {e}")
        latency = int((time.time() - start_time) * 1000)
        return 0.0, latency

    # Extract README content (if available)
    readme = getattr(repo_info, "cardData", None) or {}
    readme_text = ""
    
    # Try to construct text from metadata tags
    if readme:
        if "datasets" in readme:
            val = readme.get("datasets", [])
            if isinstance(val, list):
                readme_text += " ".join(str(v) for v in val)
            else:
                readme_text += str(val)
        if "model-index" in readme:
            # parsing model-index if needed
            pass
    
    # Check for dataset mentions in README
    dataset_documented = False
    dataset_patterns = [r"dataset", r"corpus", r"benchmark", r"train set", r"eval set"]
    for pattern in dataset_patterns:
        if re.search(pattern, readme_text, re.IGNORECASE):
            dataset_documented = True
            break
            
    # If explicit tags exist, it's documented
    if readme and ("datasets" in readme or "dataset_info" in readme):
        dataset_documented = True

    # Check for code scripts or requirements
    has_code = False
    
    if url_type == "model":
        # We have siblings from the API call
        files = [f.rfilename for f in repo_info.siblings]
        has_code = any(
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
            "eval.py"
        ]
        
        for filename in common_code_files:
            try:
                # file_exists is efficient (HEAD request)
                if HF_API.file_exists(repo_id=repo_id, filename=filename, repo_type="dataset"):
                    has_code = True
                    break
            except Exception:
                continue

    score = 0.0
    if dataset_documented:
        score += 0.5
    if has_code:
        score += 0.5

    latency = int((time.time() - start_time) * 1000)
    return round(score, 2), latency