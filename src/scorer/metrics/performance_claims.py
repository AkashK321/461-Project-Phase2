"""
Evaluate model performance claims.
"""

import json
import os
import shutil
import tempfile
import time
from typing import Optional, Tuple
import requests
import zipfile
import io
from huggingface_hub import hf_hub_download
from dotenv import load_dotenv
from pathlib import Path
import re
import logging
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an expert software engineer evaluating the performance claims "
    "of a machine learning model.\n"
    "Given the README of a model, assess the documentation for performance claims.\n"
    "Analyze the provided model documentation (README) "
    "and assign harsh scores from 0.0-1.0 that "
    "penalize missing, vague, or incomplete information. Do not ever reward absence "
    "of information. Be very strict with scoring.\n"
    "The performance claims should be judged on aspects such as "
    "detailed metrics, benchmarking results, evaluation procedures, "
    "cleanliness, relevance, and proper formatting.\n"
    "Normalize the score [0,1] based on quality indicators "
    "such as the critera mentioned above.\n"
    "Return STRICT JSON with two fields:\n"
    '{"score": <float between 0 and 1>, "rationale": "<<=200 chars explanation>"}\n\n'
    "Do NOT include anything else."
)
USER_PROMPT_TEMPLATE = "README (truncated if very long)\n----------------\n{readme}\n"


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


def get_performance_claims(url: str, url_type: str) -> Tuple[float, int]:
    """
    Function to get model or code performance claims based on URL type.
    """

    start_time = time.time()
    score = 0.0
    readme_text = ""
    logger.info(f"Evaluating performance claims for {url_type} URL: {url}")

    if url_type == "code":
        # clone GitHub repo and check readme for performance claims
        score, readme_text = _check_code_repo_performance(url)
    elif url_type == "model":
        # check Hugging face model card for performance claims
        score, readme_text = _check_model_card_performance(url)

    if score < 0.3:
        logger.info(f"Heuristic score {score} < 0.3. Querying LLM.")
        llm_score = _ask_llm(readme_text)
        if llm_score is not None:
            score = llm_score

    logger.info(f"Model performance claims score: {score}")

    latency = int((time.time() - start_time) * 1000)

    return score, latency


def _check_code_repo_performance(code_url: str) -> Tuple[float, str]:
    """
    Function to check the code repo for performance claims without using GitPython.
    Downloads the repo as a zip archive.
    """

    score = 0.0
    readme_text = ""
    temp_dir = tempfile.mkdtemp()

    try:
        # 1. Clean up URL to construct ZIP link
        if code_url.endswith(".git"):
            code_url = code_url[:-4]

        # 2. Attempt download (Try 'main' branch first, then 'master')
        branches = ["main", "master"]
        download_success = False

        for branch in branches:
            zip_url = f"{code_url}/archive/refs/heads/{branch}.zip"
            try:
                response = requests.get(zip_url)
                if response.status_code == 200:
                    # Extract to temp_dir
                    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                        z.extractall(temp_dir)
                    download_success = True
                    break
            except Exception as e:
                logger.warning(f"Failed to download branch {branch}: {e}")
                continue

        if not download_success:
            logger.warning(f"Cannot download repo archive from: {code_url}")
            return 0.0, ""

        # 3. Locate files (GitHub zips create a nested root folder)
        # We traverse the temp_dir to find content regardless of the root folder name
        found_test_script = False

        for root, dirs, files in os.walk(temp_dir):
            for filename in files:
                # Check for README
                if filename.lower() == "readme.md":
                    logger.info(f"Found README at: {os.path.join(root, filename)}")
                    try:
                        with open(
                            os.path.join(root, filename),
                            "r",
                            encoding="utf-8",
                            errors="ignore",
                        ) as f:
                            readme_text = f.read()
                    except Exception:
                        logger.warning("Cannot open readme")

                # Check for test/eval scripts
                if "test" in filename.lower() or "eval" in filename.lower():
                    found_test_script = True
                    logger.info(f"Found test script at: {os.path.join(root, filename)}")

        # 4. Calculate Score
        keywords = ["benchmark", "evaluation", "performance"]
        score = _keyword_score(readme_text, keywords)

        if found_test_script:
            score = max(score, 0.9)

    # remove the repo
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    return score, readme_text


def _check_model_card_performance(model_url: str) -> Tuple[float, str]:
    """
    Function to check the model card/README on Hugging Face for performance claims.
    """

    load_dotenv(dotenv_path=Path(__file__).resolve().parents[3] / ".env")
    hf_token = os.getenv("HF_TOKEN")

    score = 0.0
    text = ""

    try:
        # # extract repo_id from url
        # if "huggingface.co/" not in model_url:
        #     raise ValueError(f"Invalid HuggingFace URL: {model_url}")
        # model_id = model_url.split("huggingface.co/")[-1].strip("/")

        # extract repo_id from URL
        if "huggingface.co/" not in model_url:
            raise ValueError(f"Invalid HuggingFace URL: {model_url}")

        model_id = model_url.split("huggingface.co/")[-1].strip("/")

        # Remove any /tree/main, /blob/... etc
        model_id = model_id.split("/tree")[0]
        model_id = model_id.split("/blob")[0]

        logger.info(f"Extracted model ID: {model_id}")

        # download README.md from the repo
        readme_path = hf_hub_download(
            repo_id=model_id, filename="README.md", token=hf_token
        )

        # read full text
        with open(readme_path, "r", encoding="utf-8") as f:
            text = f.read()

        logger.info(f"Downloaded README.md for model ID: {model_id}")

        # define keywords to check in the readme
        sentences = re.split(r"[.!?]", text)
        keywords = [
            "benchmark",
            "evaluation",
            "performance",
            "metric",
            "score",
            "result",
            "outcome",
            "effectiveness",
            "efficacy",
            "validation",
            "accuracy",
            "f1",
            "precision",
            "recall",
            "auc",
            "roc",
            "top-1",
            "top-5",
            "mse",
            "mae",
            "rmse",
            "loss",
            "cross-entropy",
            "log-loss",
            "bleu",
            "rouge",
            "meteor",
            "perplexity",
            "iou",
            "ap",
            "map",
            "precision-recall",
            "latency",
            "throughput",
            "fps",
            "speed",
            "memory",
            "params",
            "size",
            "parameter",
            "parameters",
            "recognition",
            "beneficial",
        ]

        # Keep track of keywords that have already been counted
        counted_keywords = set()
        keyword_count = 0

        logger.info(
            "Analyzing README for performance keywords on "
            "{len(sentences)} sentences from readme."
        )

        for sent in sentences:
            sent_lower = sent.lower()
            for kw in keywords:
                # match whole word only using \b for word boundaries
                if re.search(rf"\b{re.escape(kw)}\b", sent_lower):
                    if kw not in counted_keywords:
                        # bonus if numeric value present
                        if re.search(r"\b\d+(\.\d+)?%?\b", sent_lower):
                            keyword_count += 2
                        else:
                            keyword_count += 1
                        counted_keywords.add(kw)
                        # print(kw)

        score = min(keyword_count / 5, 1.0)

        # print(f"Number of performance keywords = {keyword_count}")

    except Exception as e:
        print(f"Error checking model card: {e}")

    return round(score, 2), text


def _keyword_score(text: str, keywords: list[str]) -> float:
    """
    Function to count keywords in a string and compute score.
    """

    if not text:
        return 0.0
    text = text.lower()
    matches = 0
    for keyword in keywords:
        if keyword in text:
            matches += 1
    score = min(1.0, matches / (len(keywords) / 2))
    return score
