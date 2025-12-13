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
import logging

logger = logging.getLogger(__name__)
# Setup Auth
gh_token = os.getenv("GITHUB_TOKEN")
headers = {
    "Accept": "application/vnd.github.v3+json",
}
if gh_token:
    headers["Authorization"] = f"token {gh_token}"
else:
    logger.warning("No GITHUB_TOKEN set. Rate limits will be strict (60/hr).")

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
    logger.info(f"Running radon on {path}")
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
        except Exception as e:
            logger.debug(f"Radon command failed: {cmd} -> {e}")
            continue
    if not result:
        logger.warning("Radon execution failed or returned no output.")
        return 0.0

    score_map = {"A": 1.0, "B": 0.8, "C": 0.6, "D": 0.4, "E": 0.2, "F": 0.0}
    vals = []
    for line in result.stdout.splitlines():
        if " - " in line:
            parts = line.split(" -")
            if len(parts) >= 2:
                grade = parts[-1].strip()[:1]
                vals.append(score_map.get(grade, 0.0))
    score = sum(vals) / len(vals) if vals else 0.0
    logger.info(f"Radon score: {score}")
    return score


def run_lizard(path: str) -> Optional[Dict]:
    """Run lizard for complexity analysis."""
    logger.info(f"Running lizard on {path}")
    for cmd in ([sys.executable, "-m", "lizard", path], ["lizard", path]):
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True, timeout=20
            )
            break
        except Exception as e:
            logger.debug(f"Lizard command failed: {cmd} -> {e}")
            result = None
            continue
    if not result or result.returncode != 0:
        logger.warning("Lizard execution failed.")
        return None

    total_row = None
    for line in result.stdout.splitlines():
        if re.match(r"^\s*\d+\s+\d+", line.strip()):
            total_row = line.strip()
    if not total_row:
        logger.warning("Could not find totals in Lizard output")
        return None

    parts = total_row.split()
    if len(parts) < 6:
        logger.warning("Lizard output format unexpected")
        return None
    try:
        data = {
            "Total NLOC": float(parts[0]),
            "Avg NLOC": float(parts[1]),
            "Avg CCN": float(parts[2]),
            "Avg Tokens": float(parts[3]),
            "Function Count": int(parts[4]),
            "Warning Count": int(parts[5]),
        }
        logger.info(f"Lizard metrics: {data}")
        return data
    except ValueError:
        logger.warning("Error parsing lizard output values")
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

    score = (ccn_score + nloc_score) / 2.0
    logger.info(f"Lizard calculated score: {score} (CCN: {avg_ccn}, NLOC: {avg_nloc})")
    return score


def docstring_ratio(path: str) -> float:
    logger.info(f"Calculating docstring ratio for {path}")
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
                except Exception as e:
                    logger.debug(f"Error parsing {file}: {e}")
                    continue
    ratio = documented / total if total > 0 else 0.0
    logger.info(f"Docstring ratio: {ratio} ({documented}/{total})")
    return ratio


def parse_github_url(url: str) -> Tuple[str, str, str, str]:
    """
    Parses a GitHub URL into owner, repo, branch, and sub-path.
    Handles: https://github.com/owner/repo/tree/branch/path/to/folder
    """
    url = url.rstrip("/")
    if "github.com/" not in url:
        return None, None, "main", ""

    # Remove protocol and domain
    path_part = url.split("github.com/")[-1]
    parts = path_part.split("/")

    if len(parts) < 2:
        return None, None, "main", ""

    owner = parts[0]
    repo = parts[1]
    if repo.endswith(".git"):
        repo = repo[:-4]

    branch = "main"  # Default
    subpath = ""

    # Check if URL contains tree/BRANCH
    if len(parts) >= 4 and parts[2] == "tree":
        branch = parts[3]
        if len(parts) > 4:
            subpath = "/".join(parts[4:])

    return owner, repo, branch, subpath


def _check_code_repo_quality(code_url: str) -> float:
    logger.info(f"Checking code repo quality for: {code_url}")
    temp_dir = tempfile.mkdtemp()
    MAX_FILES = 50
    MAX_TIME = 5

    try:
        owner, repo, branch, subpath = parse_github_url(code_url)
        if not owner or not repo:
            logger.warning(f"Could not parse GitHub URL: {code_url}")
            return 0.0

        logger.info(f"Repo: {owner}/{repo}, Branch: {branch}, Path: {subpath}")

        zip_url = f"https://github.com/{owner}/{repo}/archive/refs/heads/{branch}.zip"
        logger.info(f"Attempting download from {zip_url}")
        try:
            r = requests.get(zip_url, headers=headers, stream=True, timeout=5)
            if r.status_code != 200:
                zip_url = (
                    f"https://github.com/{owner}/{repo}/archive/refs/heads/master.zip"
                )
                logger.info(f"Main failed, trying master: {zip_url}")
                r = requests.get(zip_url, headers=headers, stream=True, timeout=5)
        except Exception as e:
            logger.warning(f"Error downloading code repository zip from {zip_url}: {e}")
            return 0.0

        if not r or r.status_code != 200:
            logger.warning(
                f"Failed to download zip. Status code: {r.status_code if r else 'None'}"
            )
            return 0.0

        start_extract = time.time()
        file_count = 0
        has_python = False

        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            all_files = z.namelist()
            if not all_files:
                return 0.0

            # Identify root folder in zip (e.g. "repo-main/")
            root_in_zip = all_files[0].split("/")[0]

            # Construct the full path we want to extract
            target_prefix = f"{root_in_zip}/"
            if subpath:
                target_prefix += f"{subpath}/"
                # clean up double slashes just in case
                target_prefix = target_prefix.replace("//", "/")

            logger.info(f"Extracting files starting with: {target_prefix}")

            for file_info in z.infolist():
                if (time.time() - start_extract) > MAX_TIME:
                    break
                if file_count >= MAX_FILES:
                    break

                # Only extract if it is inside our target folder
                if file_info.filename.startswith(target_prefix):
                    # Check exclude list using relative path
                    rel_path = file_info.filename[len(target_prefix) :]
                    if not rel_path or rel_path.endswith("/"):
                        continue

                    if is_allowed(rel_path):
                        z.extract(file_info, temp_dir)
                        file_count += 1
                        if file_info.filename.endswith(".py"):
                            has_python = True

        logger.info(f"Extracted {file_count} files. Has Python: {has_python}")

        if file_count == 0:
            logger.warning("No files extracted from repo.")
            return 0.0

        current_root = os.path.join(temp_dir, root_in_zip)
        if subpath:
            current_root = os.path.join(current_root, subpath)

        scan_dir = current_root
        if not os.path.exists(scan_dir):
            scan_dir = temp_dir  # Fallback

        logger.info(f"Scanning directory: {scan_dir}")

        # Flatten
        items = os.listdir(scan_dir)
        if len(items) == 1 and os.path.isdir(os.path.join(scan_dir, items[0])):
            top_level = os.path.join(scan_dir, items[0])
            for item in os.listdir(top_level):
                shutil.move(os.path.join(top_level, item), scan_dir)
            os.rmdir(top_level)

        # 1. Reliability
        reliability = 0.0
        for root, _, files in os.walk(scan_dir):
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
        logger.info(f"Reliability score: {reliability}")

        # 2. Complexity
        radon_score = run_radon(scan_dir)
        lizard_totals = run_lizard(scan_dir)
        lizard_score = score_from_lizard_totals(lizard_totals) if lizard_totals else 0.5

        # If radon failed or no python, rely on lizard.
        # If lizard failed, assume average.
        if has_python:
            complexity = max(radon_score, lizard_score)
        else:
            complexity = lizard_score
        logger.info(
            f"Complexity score: {complexity} (Radon: {radon_score}, Lizard: {lizard_score})"
        )

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
            if os.path.exists(os.path.join(scan_dir, ci)):
                testability = 1.0
                break
        logger.info(f"Testability score: {testability}")

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
            if os.path.exists(os.path.join(scan_dir, bf)):
                portability = 1.0
                break
        logger.info(f"Portability score: {portability}")

        # 5. Reusability
        reusability = 0.0
        readme_score = 0.0
        for name in ["README.md", "README", "readme.txt"]:
            if os.path.exists(os.path.join(scan_dir, name)):
                readme_score = 0.8
                break

        if has_python:
            doc_score = docstring_ratio(scan_dir)
            reusability = max(readme_score, doc_score)
        else:
            reusability = readme_score
        logger.info(f"Reusability score: {reusability}")

        # --- BALANCED WEIGHTS ---
        final_score = (
            complexity * 0.40
            + reliability * 0.20
            + testability * 0.10
            + portability * 0.10
            + reusability * 0.20
        )

        logger.info(f"Final Code Quality Score: {final_score}")
        return min(1.0, max(0.0, final_score))

    except Exception as e:
        logger.error(f"Error in _check_code_repo_quality: {e}", exc_info=True)
        return 0.0
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def get_code_quality(url: str, url_type: str) -> Tuple[float, int]:
    start = time.time()
    logger.info(f"Starting code quality check for {url} ({url_type})")
    score = 0.0
    if url_type == "code":
        score = _check_code_repo_quality(url)
    else:
        logger.warning(f"Unsupported URL type for code quality: {url_type}")
        score = 0.5
    latency = int((time.time() - start) * 1000)
    logger.info(f"Code quality check finished. Score: {score}, Latency: {latency}ms")
    return score, latency
