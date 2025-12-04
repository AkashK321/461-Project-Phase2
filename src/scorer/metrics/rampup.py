# This code implements the ramp up feasibility metric
# by utilizing Purdue GenAI Studio (LLM prompt) via RCAC.

from __future__ import annotations

import logging
import os
import re
import time
import json
from pathlib import Path
from typing import Tuple, List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from dotenv import load_dotenv
from urllib.parse import urlparse

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

HF_API = HfApi()

# Load .env if present so GEN_AI_STUDIO_API_KEY / GENAI_* vars are picked up
load_dotenv()
logger = logging.getLogger(__name__)


def _parse_repo_id(url: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Parses a URL to determine repo_id and repo_type (model, dataset, space).
    Returns (repo_id, repo_type).
    """
    p = urlparse(url)

    # Handle direct "huggingface.co" links
    if "huggingface.co" in p.netloc:
        parts = [x for x in p.path.split("/") if x]
        if not parts:
            return None, None

        if parts[0] == "datasets" and len(parts) >= 3:
            return f"{parts[1]}/{parts[2]}", "dataset"
        elif parts[0] == "spaces" and len(parts) >= 3:
            return f"{parts[1]}/{parts[2]}", "space"
        elif len(parts) >= 2:
            return f"{parts[0]}/{parts[1]}", "model"

    # Fallback for simple strings like "user/repo"
    if url.count("/") == 1:
        return url, "model"

    return None, None


README_CANDIDATES = [
    "README.md",
    "readme.md",
    "README.rst",
    "readme.rst",
    "README",
    "Readme.md",
    "docs/README.md",
    "docs/index.md",
]
SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "dist",
    "build",
}


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


def _get_repo_tree_hf(
    repo_id: str, repo_type: str, max_files: int = 120
) -> Tuple[str, Optional[str]]:
    """
    Uses HfApi to list files. Returns (tree_string, readme_content).
    """
    try:
        files = HF_API.list_repo_files(repo_id=repo_id, repo_type=repo_type)
    except Exception:
        return "", None

    filtered_files = []
    readme_filename = None
    count = 0

    for f in files:
        # Simple skip logic matches your original SKIP_DIRS
        if any(p in SKIP_DIRS for p in f.split("/")):
            continue

        filtered_files.append(f)

        # Identify readme without reading content yet
        if readme_filename is None and f in README_CANDIDATES:
            readme_filename = f

        count += 1
        if count >= max_files * 2:
            break

    tree_str = "\n".join(filtered_files[:max_files])

    readme_content = ""
    if readme_filename:
        readme_content = _fetch_file_content(repo_id, repo_type, readme_filename)

    return tree_str, readme_content


SYSTEM_PROMPT = (
    "You are a precise software onboarding evaluator.\n"
    "Given a repository README and a brief repo file listing, "
    "rate how fast a new engineer could ramp up.\n"
    "Consider ONLY: installation clarity, prerequisites, quickstart/usage examples, "
    "runnable commands, troubleshooting,\n"
    "links to docs/tutorials, and overall coherence/structure of the README.\n\n"
    "Return STRICT JSON with two fields:\n"
    '{"score": <float between 0 and 1>, "rationale": "<<=200 chars explanation>"}\n\n'
    "Do NOT include anything else."
)
USER_PROMPT_TEMPLATE = (
    "REPO SUMMARY (first {n_files} files)\n----------------\n{tree}\n\n"
    "README (truncated if very long)\n----------------\n{readme}\n"
)

_HEUR_PATTERNS = [
    r"\bpip install\b",
    r"\bconda (?:create|install)\b",
    r"\bgit clone\b",
    r"\bpython (?:-m )?\w+\.py\b",
    r"\busage\b",
    r"\bquick\s*start\b",
    r"\bexample\b",
    r"\brequirements\.txt\b",
    r"\benvironment\.yml\b",
    r"```",
    r"\btroubleshoot",
    r"\bfaq\b",
    r"\bdocs?\b",
    r"\btutorial\b",
]


def _heuristic_rampup(readme: str, tree: str) -> float:
    txt = (readme or "") + "\n" + (tree or "")
    hits = sum(1 for pat in _HEUR_PATTERNS if re.search(pat, txt, flags=re.IGNORECASE))
    return max(0.0, min(1.0, 0.15 + 0.06 * hits))


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


def _ask_llm(readme: str, tree: str) -> Optional[float]:
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
        # strip DeepSeek R1 thinking + code fences + leading/trailing junk
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
        # salvage: look for a number in [0,1] and use that
        m = re.search(r"\b(0(?:\.\d+)?|1(?:\.0+)?)\b", txt)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                return None
        return None

    # Build the README+tree prompt
    readme = (readme or "").strip()
    if len(readme) > 20000:
        readme = readme[:20000] + "\n\n[TRUNCATED]"
    user_prompt_1 = (
        USER_PROMPT_TEMPLATE.format(n_files=120, tree=tree[:8000], readme=readme)
        + '\n\nReturn ONLY strict JSON: \
        {"score": <float 0..1>, "rationale": "<=200 chars"}.'
    )

    # Preflight debug
    payload = _build_payload(user_prompt_1)
    payload_str = json.dumps(payload)
    # Send with retries

    logger.info(f"Payload: {payload_str}")

    session = _session_with_retry()
    try:
        resp = session.post(
            url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            data=payload_str,
            timeout=90,
        )
        logger.info(f"Response: {resp}")
    except requests.exceptions.Timeout:
        return None
    except requests.exceptions.SSLError:
        return None
    except requests.exceptions.ConnectionError:
        return None
    except Exception:
        return None

    # HTTP status handling
    if resp.status_code in (401, 402, 403):
        return None
    if resp.status_code == 400 and "Model not found" in (resp.text or ""):
        return None
    if resp.status_code != 200:
        return None

    # Parse 1st pass
    try:
        data = resp.json()
    except ValueError:
        return None

    txt = None
    try:
        txt = data["choices"][0]["message"]["content"]
    except Exception:
        txt = data.get("output_text") or data.get("text") or ""
    score = _parse_or_salvage(txt)
    if isinstance(score, float):
        return max(0.0, min(1.0, score))

    # Second pass: ultra-strict re-ask (short, no repo text again)
    user_prompt_2 = (
        "Output EXACTLY this JSON (no analysis, no extra keys, no markdown): "
        '{"score": <float 0..1>, "rationale": "<=200 chars>"}'
    )
    payload2 = _build_payload(user_prompt_2)
    logger.info(f"Second pass payload: {json.dumps(payload2)}")
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
        logger.info(f"Second pass response: {resp2}")
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


def get_ramp_up(url: str, url_type: str) -> Tuple[float, int]:
    start = time.time()
    try:
        # --- CHANGED: URL Parsing ---
        repo_id, parsed_type = _parse_repo_id(url)

        # Determine strict type if provided, else use parsed
        if url_type and url_type.lower() in ["model", "dataset", "space"]:
            final_type = url_type.lower()
        else:
            final_type = parsed_type if parsed_type else "model"

        if not repo_id:
            return 0.0, int((time.time() - start) * 1000)

        # --- CHANGED: Retrieve Data via API instead of Clone ---
        tree, readme = _get_repo_tree_hf(repo_id, final_type, max_files=120)

        llm_score = _ask_llm(readme, tree)

        if llm_score is not None:
            # Clamp LLM score to valid range
            llm_score = max(0.0, min(1.0, float(llm_score)))

            return llm_score, int((time.time() - start) * 1000)

        # Fallback heuristic
        score = _heuristic_rampup(readme, tree)
        base_score = score

        readme_text = readme or ""
        readme_len = len(readme_text)

        length_boost = min(0.10, (readme_len / 15000) * 0.10)

        doc_keywords = [
            "install",
            "installation",
            "usage",
            "example",
            "quickstart",
            "tutorial",
            "guide",
            "requirements",
            "pip",
            "conda",
        ]
        hits = sum(1 for k in doc_keywords if k in readme_text.lower())
        doc_boost = min(0.15, hits * 0.03)

        tree_lines = tree.split("\n")
        structure_boost = 0.05 if len(tree_lines) > 15 else 0.0
        tiny_penalty = -0.20 if len(tree_lines) < 6 else 0.0

        raw = score + length_boost + doc_boost + structure_boost + tiny_penalty

        # Diminishing returns only on heuristic path
        score = raw * (1 - 0.15 * raw)

        # Don't let it drop below heuristic minimum
        if score < base_score:
            score = base_score

        score = max(0.0, min(1.0, score))

        return float(score), int((time.time() - start) * 1000)

    except Exception:
        return 0.0, int((time.time() - start) * 1000)
