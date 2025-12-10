"""
Evaluates the quality of the code with balanced weights and language support.
"""

import os
import shutil
import tempfile
import time
import subprocess
import sys
import requests
import zipfile
import io
from typing import Tuple, Dict, Optional
import ast
import re

# --- FILTERS ---
EXCLUDED_EXTS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".tiff",
    ".svg",
    ".webp",
    ".mp4",
    ".mov",
    ".avi",
    ".mp3",
    ".wav",
    ".zip",
    ".tar",
    ".gz",
    ".7z",
    ".rar",
    ".pdf",
    ".doc",
    ".docx",
    ".ppt",
    ".pptx",
    ".xls",
    ".xlsx",
    ".txt",
    ".pyc",
    ".pyo",
    ".pyd",
    ".o",
    ".obj",
    ".dll",
    ".exe",
    ".so",
    ".dylib",
    ".class",
    ".jar",
    ".bin",
    ".onnx",
    ".pb",
    ".h5",
    ".hdf5",
    ".safetensors",
    ".pack",
    ".idx",
    ".sample",
}

EXCLUDED_DIRS = {
    "__pycache__",
    ".git",
    ".github",
    ".idea",
    ".vscode",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "bin",
    "obj",
    "build",
    "dist",
    "target",
    "t",
    "test",
    "tests",
    "testing",
    "spec",
    "specs",
    "doc",
    "docs",
    "documentation",
    "po",
    "site",
    "examples",
    "samples",
    "contrib",
}


def is_allowed(filename: str) -> bool:
    parts = filename.split("/")
    if any(p.startswith(".") for p in parts):
        return False
    for part in parts:
        if part.lower() in EXCLUDED_DIRS:
            return False
    _, ext = os.path.splitext(filename)
    if ext.lower() in EXCLUDED_EXTS:
        return False
    return True


def run_radon(path: str) -> float:
    """Run radon for Python maintainability."""
    cmds = [
        [sys.executable, "-m", "radon", "mi", "-s", path],
        ["radon", "mi", "-s", path],
    ]
    result = None
    for cmd in cmds:
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True, timeout=20
            )
            break
        except Exception:
            continue
    if not result:
        return 0.0

    score_map = {"A": 1.0, "B": 0.8, "C": 0.6, "D": 0.4, "E": 0.2, "F": 0.0}
    vals = []
    for line in result.stdout.splitlines():
        if " - " in line:
            parts = line.split(" -")
            if len(parts) >= 2:
                grade = parts[-1].strip()[:1]
                vals.append(score_map.get(grade, 0.0))
    return sum(vals) / len(vals) if vals else 0.0


def run_lizard(path: str) -> Optional[Dict]:
    """Run lizard for complexity analysis."""
    for cmd in ([sys.executable, "-m", "lizard", path], ["lizard", path]):
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True, timeout=20
            )
            break
        except Exception:
            result = None
            continue
    if not result or result.returncode != 0:
        return None

    total_row = None
    for line in result.stdout.splitlines():
        if re.match(r"^\s*\d+\s+\d+", line.strip()):
            total_row = line.strip()
    if not total_row:
        return None

    parts = total_row.split()
    if len(parts) < 6:
        return None
    try:
        return {
            "Total NLOC": float(parts[0]),
            "Avg NLOC": float(parts[1]),
            "Avg CCN": float(parts[2]),
            "Avg Tokens": float(parts[3]),
            "Function Count": int(parts[4]),
            "Warning Count": int(parts[5]),
        }
    except ValueError:
        return None


def score_from_lizard_totals(totals: dict) -> float:
    if not totals:
        return 0.0

    avg_ccn = totals.get("Avg CCN", 0)
    if avg_ccn <= 10:
        ccn_score = 1.0
    elif avg_ccn <= 15:
        ccn_score = 0.8
    elif avg_ccn <= 25:
        ccn_score = 0.5
    else:
        ccn_score = 0.2

    avg_nloc = totals.get("Avg NLOC", 0)
    if avg_nloc <= 40:
        nloc_score = 1.0
    elif avg_nloc <= 60:
        nloc_score = 0.8
    elif avg_nloc <= 120:
        nloc_score = 0.5
    else:
        nloc_score = 0.2

    return (ccn_score + nloc_score) / 2.0


def docstring_ratio(path: str) -> float:
    total = 0
    documented = 0
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith(".py"):
                try:
                    with open(
                        os.path.join(root, file), "r", encoding="utf-8", errors="ignore"
                    ) as fh:
                        tree = ast.parse(fh.read())
                        for node in ast.walk(tree):
                            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                                total += 1
                                if ast.get_docstring(node):
                                    documented += 1
                except Exception:
                    continue
    return documented / total if total > 0 else 0.0


def _check_code_repo_quality(code_url: str) -> float:
    temp_dir = tempfile.mkdtemp()
    MAX_FILES = 150
    MAX_TIME = 10.0

    try:
        if "github.com" in code_url:
            repo_path = code_url.split("github.com/")[-1].strip("/")
            if repo_path.endswith(".git"):
                repo_path = repo_path[:-4]
        else:
            repo_path = code_url.split("/")[-1]

        zip_url = f"https://github.com/{repo_path}/archive/refs/heads/main.zip"
        try:
            r = requests.get(zip_url, stream=True, timeout=15)
            if r.status_code != 200:
                zip_url = (
                    f"https://github.com/{repo_path}/archive/refs/heads/master.zip"
                )
                r = requests.get(zip_url, stream=True, timeout=15)
        except Exception:
            return 0.0

        if not r or r.status_code != 200:
            return 0.0

        start_extract = time.time()
        file_count = 0
        has_python = False

        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            for file_info in z.infolist():
                if (time.time() - start_extract) > MAX_TIME:
                    break
                if file_count >= MAX_FILES:
                    break

                if not file_info.filename.endswith("/") and is_allowed(
                    file_info.filename
                ):
                    z.extract(file_info, temp_dir)
                    file_count += 1
                    if file_info.filename.endswith(".py"):
                        has_python = True

        if file_count == 0:
            return 0.0

        # Flatten
        items = os.listdir(temp_dir)
        if len(items) == 1 and os.path.isdir(os.path.join(temp_dir, items[0])):
            top_level = os.path.join(temp_dir, items[0])
            for item in os.listdir(top_level):
                shutil.move(os.path.join(top_level, item), temp_dir)
            os.rmdir(top_level)

        # 1. Reliability
        reliability = 0.0
        for root, _, files in os.walk(temp_dir):
            if "test" in root.lower():
                reliability = 0.7
            for f in files:
                if "test" in f.lower():
                    reliability = 0.7
                if f.lower() in [
                    "makefile",
                    "cmakelists.txt",
                    "pom.xml",
                    "build.gradle",
                ]:
                    reliability = max(reliability, 0.7)
        if reliability == 0.0:
            reliability = 0.3  # Base reliability for existing code

        # 2. Complexity
        radon_score = run_radon(temp_dir)
        lizard_totals = run_lizard(temp_dir)
        lizard_score = score_from_lizard_totals(lizard_totals) if lizard_totals else 0.5

        # If radon failed or no python, rely on lizard.
        # If lizard failed, assume average.
        if has_python:
            complexity = max(radon_score, lizard_score)
        else:
            complexity = lizard_score

        # 3. Testability
        testability = 0.0
        ci_files = [
            ".github",
            ".gitlab-ci.yml",
            "azure-pipelines.yml",
            ".travis.yml",
            "circle.yml",
        ]
        for ci in ci_files:
            if os.path.exists(os.path.join(temp_dir, ci)):
                testability = 1.0
                break

        # 4. Portability (Language Agnostic)
        portability = 0.0
        build_files = [
            "Dockerfile",
            "docker-compose.yml",
            "requirements.txt",
            "setup.py",
            "package.json",
            "Makefile",
            "CMakeLists.txt",
            "pom.xml",
            "build.gradle",
            "Cargo.toml",
            "go.mod",
        ]
        for bf in build_files:
            if os.path.exists(os.path.join(temp_dir, bf)):
                portability = 1.0
                break

        # 5. Reusability
        reusability = 0.0
        readme_score = 0.0
        for name in ["README.md", "README", "readme.txt"]:
            if os.path.exists(os.path.join(temp_dir, name)):
                readme_score = 0.8
                break

        if has_python:
            doc_score = docstring_ratio(temp_dir)
            reusability = max(readme_score, doc_score)
        else:
            reusability = readme_score

        # --- BALANCED WEIGHTS ---
        final_score = (
            complexity * 0.40
            + reliability * 0.20
            + testability * 0.10
            + portability * 0.10
            + reusability * 0.20
        )

        return min(1.0, max(0.0, final_score))

    except Exception:
        return 0.0
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def get_code_quality(url: str, url_type: str) -> Tuple[float, int]:
    start = time.time()
    score = 0.0
    if url_type == "code":
        score = _check_code_repo_quality(url)
    return score, int((time.time() - start) * 1000)
