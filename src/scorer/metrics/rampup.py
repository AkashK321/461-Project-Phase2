# This code implements the ramp up feasibility metric
# by utilizing Purdue GenAI Studio (LLM prompt) via RCAC.

from __future__ import annotations

import os
import re
import time
import json
import shutil
import tempfile
from pathlib import Path
from typing import Tuple, List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from git import Repo
from dotenv import load_dotenv
from urllib.parse import urlparse

# Load .env if present so GEN_AI_STUDIO_API_KEY / GENAI_* vars are picked up
load_dotenv()


def _to_clone_url(url: str, url_type: str) -> str:
    p = urlparse(url)
    host = p.netloc.lower()
    parts = [x for x in p.path.split("/") if x]

    if "github.com" in host:
        if len(parts) < 2:
            raise ValueError("GitHub URL must be /owner/repo")
        return f"https://github.com/{parts[0]}/{parts[1]}.git"

    if "huggingface.co" in host:
        if parts and parts[0].lower() == "datasets":
            if len(parts) < 3:
                raise ValueError("HF dataset URL must be /datasets/<ns>/<name>")
            repo_id = f"{parts[1]}/{parts[2]}"
            return f"https://huggingface.co/datasets/{repo_id}"
        else:
            if len(parts) < 2:
                raise ValueError("HF model URL must be /<ns>/<name>")
            repo_id = f"{parts[0]}/{parts[1]}"
            return f"https://huggingface.co/{repo_id}"

    return url


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


def _read_first_readme(repo_dir: str) -> str:
    for rel in README_CANDIDATES:
        p = Path(repo_dir) / rel
        if p.exists() and p.is_file():
            try:
                return p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                pass
    return ""


def _top_level_summary(repo_dir: str, max_files: int = 120) -> str:
    entries: List[str] = []
    root = Path(repo_dir)
    count = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        rel_dir = os.path.relpath(dirpath, root)
        for f in filenames:
            rel = os.path.normpath(os.path.join(rel_dir, f))
            if rel.startswith("./"):
                rel = rel[2:]
            entries.append(rel)
            count += 1
            if count >= max_files:
                break
        if count >= max_files:
            break
    return "\n".join(entries)


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

    score2 = _parse_or_salvage(txt2)
    if isinstance(score2, float):
        return max(0.0, min(1.0, score2))

    return None


def get_ramp_up(url: str, url_type: str) -> Tuple[float, int]:
    start = time.time()
    temp_dir = tempfile.mkdtemp()
    try:
        kind = (
            "dataset"
            if url_type.lower() == "dataset"
            else "model" if url_type.lower() == "model" else "code"
        )
        clone_url = _to_clone_url(url, kind)
        env = os.environ.copy()
        env.setdefault("GIT_LFS_SKIP_SMUDGE", "1")
        Repo.clone_from(clone_url, temp_dir, multi_options=["--depth=100"], env=env)

        readme = _read_first_readme(temp_dir)
        tree = _top_level_summary(temp_dir, max_files=120)

        score = _ask_llm(readme, tree)
        if score is None:
            score = _heuristic_rampup(readme, tree)

        readme_text = readme or ""
        readme_len = len(readme_text)

        # README length bonus: up to +0.10
        length_boost = min(0.10, (readme_len / 15000) * 0.10)

        # Documentation keyword bonus: up to +0.15
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

        # Repo structure bonus: up to +0.05
        tree_lines = tree.split("\n")
        structure_boost = 0.05 if len(tree_lines) > 15 else 0.0

        # Tiny repo penalty: up to -0.20
        tiny_penalty = -0.20 if len(tree_lines) < 6 else 0.0

        # Apply combined adjustments
        score = score + length_boost + doc_boost + structure_boost + tiny_penalty
        score = max(0.0, min(1.0, score))

        latency_ms = int((time.time() - start) * 1000)
        return float(score), latency_ms

    except Exception:
        return 0.0, int((time.time() - start) * 1000)

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
