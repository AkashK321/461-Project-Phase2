import logging
import os
import re
import json
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

SYSTEM_PROMPT = (
    "You are a precise software dependency linker.\n"
    "Your task is to identify if any of the provided candidate URLs (Datasets or Code repositories) "
    "are the correct source or training data for the given Model.\n"
    "You will be given the Model Name, a snippet of its README, and a list of Candidate URLs.\n"
    "Analyze the README for mentions of the candidate names or links.\n"
    "Return STRICT JSON with two fields:\n"
    '{"matched_code_url": <string or null>, "matched_dataset_url": <string or null>}\n'
    "Only return a URL if you are confident it is the correct match based on the README or strong name similarity."
)

def _session_with_retry() -> requests.Session:
    s = requests.Session()
    r = Retry(
        total=3, backoff_factor=0.5, status_forcelist=(429, 500, 502, 503, 504)
    )
    s.mount("https://", HTTPAdapter(max_retries=r))
    return s

def match_artifacts(model_name: str, model_readme: str, candidates: list) -> dict:
    """
    Matches a model to a code repo and dataset repo from a list of candidates.
    
    :param model_name: Name of the model (e.g. 'bert-base-uncased')
    :param model_readme: Content of the model's README
    :param candidates: List of dicts, each having {'url': str, 'type': 'code'|'dataset', 'name': str}
    :return: {'matched_code_url': str|None, 'matched_dataset_url': str|None}
    """
    logger.info(f"--- [Matcher] Running for Model: {model_name} ---")
    logger.info(f"--- [Matcher] Candidate count: {len(candidates)} ---")
    
    # 1. Heuristic: Strict URL/Name matching
    # If the model README explicitly links to a candidate URL, it's a very strong signal.
    matched_code = None
    matched_dataset = None
    
    candidates_str = ""
    for idx, cand in enumerate(candidates):
        url = cand.get('url', '')
        # Handle cases where DB might use different keys or the dictionary structure varies
        c_name = cand.get('model_name') or cand.get('name') or "Unknown"
        c_type = cand.get('type', 'unknown')
        
        candidates_str += f"{idx}. [{c_type}] {c_name} ({url})\n"
        
        # Simple check: is the URL in the readme?
        if url and model_readme and url in model_readme:
            logger.info(f"--- [Matcher] Found direct link in README to {url} ({c_type})")
            if c_type == 'code':
                matched_code = url
            elif c_type == 'dataset':
                matched_dataset = url

    # If we found both via direct links, return immediately
    if matched_code and matched_dataset:
        logger.info(f"--- [Matcher] Result for {model_name}: Code={matched_code}, Dataset={matched_dataset}")
        return {"matched_code_url": matched_code, "matched_dataset_url": matched_dataset}

    # 2. LLM Matching
    # If we are missing matches, ask the LLM to infer based on context
    api_key = os.getenv("GEN_AI_STUDIO_API_KEY", "").strip()
    if not api_key:
        logger.warning("--- [Matcher] No GEN_AI_STUDIO_API_KEY, skipping LLM matching.")
        return {"matched_code_url": matched_code, "matched_dataset_url": matched_dataset}

    base = os.getenv("GENAI_BASE_URL", "https://genai.rcac.purdue.edu").rstrip("/")
    path = os.getenv("GENAI_PATH", "/api/chat/completions")
    url = f"{base}{path}"
    model = os.getenv("GENAI_MODEL", "").strip() or "deepseek-r1:7b"
    
    # Truncate readme
    readme_snippet = (model_readme or "")[:5000] 
    
    user_prompt = (
        f"MODEL NAME: {model_name}\n"
        f"README SNIPPET:\n{readme_snippet}\n\n"
        f"CANDIDATES:\n{candidates_str}\n\n"
        "Return the JSON for matched_code_url and matched_dataset_url."
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }

    try:
        session = _session_with_retry()
        resp = session.post(
            url,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=30
        )
        if resp.status_code == 200:
            data = resp.json()
            content = ""
            if "choices" in data and len(data["choices"]) > 0:
                content = data["choices"][0]["message"]["content"]
            
            # Clean content (strip markdown code blocks if present)
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
            content = re.sub(r"```json", "", content).replace("```", "").strip()
            
            try:
                parsed = json.loads(content)
                
                # Prefer LLM match if we didn't have a direct link match
                if not matched_code and parsed.get("matched_code_url"):
                    matched_code = parsed.get("matched_code_url")
                    logger.info(f"--- [Matcher] LLM matched Code: {matched_code}")
                    
                if not matched_dataset and parsed.get("matched_dataset_url"):
                    matched_dataset = parsed.get("matched_dataset_url")
                    logger.info(f"--- [Matcher] LLM matched Dataset: {matched_dataset}")

            except json.JSONDecodeError:
                logger.warning(f"--- [Matcher] Failed to parse LLM JSON: {content[:100]}...")

    except Exception as e:
        logger.error(f"--- [Matcher] Error in LLM matching: {e}")

    logger.info(f"--- [Matcher] Final Result for {model_name}: Code={matched_code}, Dataset={matched_dataset}")
    return {"matched_code_url": matched_code, "matched_dataset_url": matched_dataset}