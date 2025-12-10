"""
Implementing dataset quality metric scoring.
- For datasets: looks at the number of downloads and likes.
- For models: queries an LLM to evaluate dataset quality claims in the model card.
"""

import os
import time
import logging
import json
import re
from typing import Tuple, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv
from huggingface_hub import HfApi, login
from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError
from huggingface_hub import hf_hub_download

from .base import get_repo_id
import math

logger = logging.getLogger(__name__)

load_dotenv()

# HF_TOKEN = os.getenv("HF_Token")
HF_API = HfApi()
# login(token=HF_TOKEN)

# Downloads and likes targets for top tier quality
max_downloads = 1000000  # 1 million downloads
max_likes = 2000

SYSTEM_PROMPT = (
    "You are an expert data scientist evaluating the quality "
    "of datasets used in machine learning models.\n"
    "Given the README of a model, assess the quality of the dataset(s) it uses.\n"
    "Analyze the provided model documentation (README) "
    "and assign harsh scores from 0.0-1.0 that "
    "penalize missing, vague, or incomplete information. Do not ever reward absence and "
    "be very strict with increasing the score.\n"
    "If a valid dataset is found, proceed with evaluation, "
    "otherwise return a score of 0.0.\n"
    "The dataset quality should be judged based on size, "
    "completeness, labels, license, "
    "cleanliness, relevance, and proper formatting.\n"
    "Normalize the score [0,1] based on quality indicators "
    "such as the critera mentioned above.\n"
    "Return STRICT JSON with two fields:\n"
    '{"score": <float between 0 and 1>, "rationale": "<<=200 chars explanation>"}\n\n'
    "Do NOT include anything else."
)
USER_PROMPT_TEMPLATE = "README (truncated if very long)\n----------------\n{readme}\n"


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


def _fetch_file_content(repo_id: str, repo_type: str, filename: str) -> str:
    """Downloads a specific file from HF Hub into memory."""
    try:
        local_path = hf_hub_download(
            repo_id=repo_id, filename=filename, repo_type=repo_type, local_dir=None
        )
        with open(local_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except (EntryNotFoundError, RepositoryNotFoundError, Exception):
        return ""


def _session_with_retry() -> requests.Session:
    logger.info("Creating session with retries")
    s = requests.Session()
    r = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["POST"]),
    )
    s.mount("https://", HTTPAdapter(max_retries=r))
    s.mount("http://", HTTPAdapter(max_retries=r))
    return s


def _extract_json_first(s: str) -> dict | None:
    if not s:
        return None
    depth = 0
    start = -1
    in_str = False
    esc = False
    quote = ""
    for i, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == quote:
                in_str = False
            continue
        if ch == '"' or ch == "'":
            in_str = True
            quote = ch
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start != -1:
                    snippet = s[start : i + 1]
                    try:
                        return json.loads(snippet)
                    except Exception:
                        start = -1
                        continue
    return None


SYSTEM_PROMPT = ""
USER_PROMPT_TEMPLATE = "{readme}"


def _ask_llm(readme: str) -> Optional[float]:
    api_key = os.getenv("GEN_AI_STUDIO_API_KEY", "").strip()
    if not api_key:
        return None

    base = os.getenv("GENAI_BASE_URL", "https://genai.rcac.purdue.edu").rstrip("/")
    path = os.getenv("GENAI_PATH", "/api/chat/completions")
    url = f"{base}{path}"
    model = os.getenv("GENAI_MODEL", "").strip() or "deepseek-r1:7b"

    def _build_payload(user_prompt: str):
        return {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": 220,
            "temperature": 0.0,
            "top_p": 1.0,
            "response_format": {"type": "json_object"},
        }

    def _clean_text(txt: str) -> str:
        if not txt:
            return ""
        txt = re.sub(r"<think>.*?</think>", "", txt, flags=re.DOTALL)
        txt = re.sub(r"^```(?:json)?\s*|\s*```$", "", txt.strip(), flags=re.IGNORECASE)
        return txt.strip()

    def _parse_or_salvage(txt: str) -> Optional[float]:
        txt = _clean_text(txt)
        parsed = _extract_json_first(txt)
        if parsed and isinstance(parsed, dict) and "score" in parsed:
            try:
                return float(parsed["score"])
            except Exception:
                pass
        m = re.search(r"\b(0(?:\.\d+)?|1(?:\.0+)?)\b", txt)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                return None
        return None

    readme = (readme or "").strip()
    if len(readme) > 20000:
        readme = readme[:20000] + "\n\n[TRUNCATED]"
    user_prompt = USER_PROMPT_TEMPLATE.format(readme=readme)

    payload = _build_payload(user_prompt)
    session = _session_with_retry()
    try:
        resp = session.post(
            url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            data=json.dumps(payload),
            timeout=90,
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        score = _parse_or_salvage(content)
        if isinstance(score, float):
            return max(0.0, min(1.0, score))
        else:
            # Second pass: ultra-strict re-ask (short, no repo text again)
            user_prompt_2 = (
                "Output EXACTLY this JSON (no analysis, no extra keys, no markdown): "
                '{"score": <float 0..1>, "rationale": "<=200 chars>"}'
            )
            payload2 = _build_payload(user_prompt_2)
            logger.info(f"Dataset quality Second pass payload: {json.dumps(payload2)}")
            try:
                resp2 = session.post(
                    url,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    data=json.dumps(payload2),
                    timeout=60,
                )
                logger.info(f"Dataset quality Second pass response: {resp2}")
            except Exception:
                return None

            if resp2.status_code != 200:
                return None

            try:
                data2 = resp2.json()
            except Exception:
                return None

            txt2 = None
            try:
                txt2 = data2["choices"][0]["message"]["content"]
            except Exception:
                txt2 = data2.get("output_text") or data2.get("text") or ""

            logger.info(f"Second pass content: {txt2}")

            score2 = _parse_or_salvage(txt2)
            logger.info(f"Second pass extracted score: {score2}")
            if isinstance(score2, float):
                return max(0.0, min(1.0, score2))

            return None
    except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
        logger.error(f"LLM query failed: {e}")
        return None


def get_dataset_quality_score(url: str, url_type: str) -> Tuple[Optional[float], int]:
    _maybe_login()
    start_time = time.time()
    logger.info(f"Starting dataset quality scoring for {url}")

    # Get repo id
    try:
        repo_id = get_repo_id(url, url_type)
        logger.info(f"Repo ID for {url} is {repo_id}")
    except Exception as e:
        logger.error(f"Error getting repo id for {url}: {e}")
        latency = int((time.time() - start_time) * 1000)
        return None, latency

    if url_type == "model":
        readme_content = _fetch_file_content(repo_id, "model", "README.md")
        if not readme_content:
            logger.warning(f"No README found for model {repo_id}, cannot use LLM.")
            return 0.0, int((time.time() - start_time) * 1000)
        logger.info(
            f"Dataset quality Fetched README content for model "
            f"{repo_id}, length {len(readme_content)}"
        )
        llm_score = _ask_llm(readme_content)
        if llm_score is not None:
            logger.info(
                f"LLM-based dataset quality score for model {repo_id}: {llm_score}"
            )
            return llm_score, int((time.time() - start_time) * 1000)
        else:
            logger.warning(f"LLM query failed for model {repo_id}, returning 0.")
            return 0.0, int((time.time() - start_time) * 1000)

    elif url_type == "dataset":
        try:
            dataset_info = HF_API.dataset_info(repo_id=repo_id, files_metadata=False)
        except Exception as e:
            logger.error(f"Could not fetch dataset info for {repo_id}: {e}")
            return 0.0, int((time.time() - start_time) * 1000)

        downloads = getattr(dataset_info, "downloads", 0) or 0
        likes = (
            getattr(dataset_info, "likes", 0) or getattr(dataset_info, "stars", 0) or 0
        )
        logger.info(f"Dataset {repo_id} has {downloads} downloads and {likes} likes.")

        downloads_score = normalize(downloads, max_downloads)
        likes_score = normalize(likes, max_likes)

        # Weighted sum of downloads and likes scores
        score = 0.8 * downloads_score + 0.2 * likes_score
        logger.info(f"Final dataset quality score for {repo_id}: {score:.2f}")
        return round(score, 2), int((time.time() - start_time) * 1000)

    else:
        logger.warning(
            f"Dataset quality score is not applicable to url_type '{url_type}'"
        )
        return None, int((time.time() - start_time) * 1000)
