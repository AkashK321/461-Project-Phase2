"""
Evaluates the quality of the code.
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


def run_radon(path: str) -> float:
    """
    Function to run radon, a Python tool that analyzes source code complexity and
    maintainability.
    """
    cmds = [
        [sys.executable, "-m", "radon", "mi", "-s", path],  # preferred
        ["radon", "mi", "-s", path],  # fallback
    ]

    result = None
    for cmd in cmds:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            break
        except FileNotFoundError:
            continue
        except subprocess.CalledProcessError:
            return 0.0

    if result is None:
        return 0.0

    score_map = {"A": 1.0, "B": 0.8, "C": 0.6, "D": 0.4, "E": 0.2, "F": 0.0}
    vals = []
    for line in result.stdout.splitlines():
        if " - " in line:
            grade = line.split(" -")[-1].strip()[:1]
            vals.append(score_map.get(grade, 0.0))
    return sum(vals) / len(vals) if vals else 0.0


def run_lizard(path: str) -> Optional[Dict]:
    """
    Function to run lizard, a multi-language code analysis tool that analyzes function
    complexity.
    """

    for cmd in ([sys.executable, "-m", "lizard", path], ["lizard", path]):
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            break
        except FileNotFoundError:
            result = None
            continue
        except subprocess.CalledProcessError:
            return None
    if result is None:
        return None

    if result.returncode != 0:
        return None

    # find the last summary row using regex
    total_row = None
    for line in result.stdout.splitlines():
        # match a line that starts with spaces then numbers/floats
        if re.match(r"^\s*\d+", line.strip()):
            total_row = line.strip()

    if not total_row:
        # print("Could not find totals in Lizard output")
        return None

    # extract numeric values from the totals row
    parts = total_row.split()
    if len(parts) < 8:
        # print("Unexpected totals format:", total_row)
        return None

    totals = {
        "Total NLOC": float(parts[0]),
        "Avg NLOC": float(parts[1]),
        "Avg CCN": float(parts[2]),
        "Avg Tokens": float(parts[3]),
        "Function Count": int(parts[4]),
        "Warning Count": int(parts[5]),
        "Function Rate": float(parts[6]),
        "NLOC Rate": float(parts[7]),
    }
    return totals


def score_from_lizard_totals(totals: dict) -> float:
    """
    Function to calculate final score from dictionary returned by run_lizard().
    """

    if not totals:
        return 0.0

    # metric 1: Cyclomatic complexity
    avg_ccn = totals.get("Avg CCN", 0)
    if avg_ccn <= 5:
        ccn_score = 1.0
    elif avg_ccn <= 10:
        ccn_score = 0.8
    elif avg_ccn <= 20:
        ccn_score = 0.5
    else:
        ccn_score = 0.2

    # metric 2: Average NLOC (function size)
    avg_nloc = totals.get("Avg NLOC", 0)
    if avg_nloc <= 30:
        nloc_score = 1.0
    elif avg_nloc <= 40:
        nloc_score = 0.8
    elif avg_nloc <= 100:
        nloc_score = 0.5
    else:
        nloc_score = 0.2

    # metric 3: Warnings
    warnings = totals.get("Warning Count", 0)
    if warnings == 0:
        warning_score = 1.0
    elif warnings <= 2:
        warning_score = 0.7
    elif warnings <= 5:
        warning_score = 0.4
    else:
        warning_score = 0.1

    # define weighted score
    weights = [0.5, 0.3, 0.2]
    components = [ccn_score, nloc_score, warning_score]
    final_score = sum(w * c for w, c in zip(weights, components)) / sum(weights)

    return final_score


def docstring_ratio(path: str) -> float:
    """
    Function to count how many Python functions or classes have docstrings.
    """
    total = 0
    documented = 0
    score = 0

    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith(".py"):
                with open(
                    os.path.join(root, file), "r", encoding="utf-8", errors="ignore"
                ) as fh:
                    try:
                        tree = ast.parse(fh.read())
                        for node in ast.walk(tree):
                            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                                total += 1
                                if ast.get_docstring(node):
                                    documented += 1
                    except Exception:
                        continue

    if total == 0:
        return 0.0
    score = documented / total
    return score


def _check_code_repo_quality(code_url: str) -> float:
    """
    Function to analyze the quality of the code.
    Uses 'requests' and 'zipfile' to download code without git binary.
    """
    
    # Define filters
    EXCLUDED_EXTS = {
        '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.svg', '.webp',
        '.mp4', '.mov', '.avi', '.mp3', '.wav',
        '.zip', '.tar', '.gz', '.7z', '.rar',
        '.pdf', '.doc', '.docx', '.ppt', '.pptx', '.xls', '.xlsx',
        '.pyc', '.pyo', '.pyd', '.o', '.obj', '.dll', '.exe', '.so', '.dylib', '.class', '.jar',
        '.DS_Store', '.bin', '.onnx', '.pb', '.h5', '.hdf5'
    }
    EXCLUDED_DIRS = {
        '__pycache__', '.git', '.idea', '.vscode', '.venv', 'venv', 'env', 'node_modules'
    }

    def is_allowed(filename):
        parts = filename.split('/')
        for part in parts:
            if part in EXCLUDED_DIRS:
                return False
        _, ext = os.path.splitext(filename)
        return ext.lower() not in EXCLUDED_EXTS

    temp_dir = tempfile.mkdtemp()
    try:
        # --- DOWNLOAD LOGIC ---
        try:
            if "github.com" in code_url:
                repo_path = code_url.split("github.com/")[-1].strip("/")
                if repo_path.endswith(".git"):
                    repo_path = repo_path[:-4]
            else:
                repo_path = code_url.split("/")[-1]

            zip_url = f"https://github.com/{repo_path}/archive/refs/heads/main.zip"
            # Add timeout to prevent hanging
            try:
                r = requests.get(zip_url, stream=True, timeout=10)
            except requests.RequestException:
                # Fallback to master
                 r = requests.Response() # dummy
                 r.status_code = 404

            if r.status_code != 200:
                zip_url = f"https://github.com/{repo_path}/archive/refs/heads/master.zip"
                r = requests.get(zip_url, stream=True, timeout=10)

            if r.status_code != 200:
                print(f"Failed to download zip from {code_url}")
                return 0.0

            # 4. Extract with FILTERING
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                for file_info in z.infolist():
                    if not file_info.filename.endswith('/') and is_allowed(file_info.filename):
                        z.extract(file_info, temp_dir)

            # 5. Flatten Directory structure
            items = os.listdir(temp_dir)
            if len(items) == 1 and os.path.isdir(os.path.join(temp_dir, items[0])):
                top_level = os.path.join(temp_dir, items[0])
                for item in os.listdir(top_level):
                    shutil.move(os.path.join(top_level, item), temp_dir)
                os.rmdir(top_level)

        except Exception as e:
            print(f"Cannot download/extract repo for code quality check: {e}")
            return 0.0

        # --- END DOWNLOAD LOGIC ---

        # first reliability check - check for the word test in the files
        reliability = 0.0
        for _, _, files in os.walk(temp_dir):
            for file in files:
                if "test" in file.lower():
                    reliability = 0.7
                    break
        # second reliability check - check for testing frameworks
        file_names = os.listdir(temp_dir)
        file_names_str = " ".join(file_names).lower()

        frameworks = ["pytest", "unittest", "mocha", "jest"]
        found_framework = False

        for fw in frameworks:
            if fw in file_names_str:
                found_framework = True
                break

        if found_framework:
            reliability = 1.0

        # check complexity (number of files and classes, complexity, etc)
        radon_score = run_radon(temp_dir)
        lizard_totals = run_lizard(temp_dir)
        lizard_score = 0.0
        if lizard_totals:
            lizard_score = score_from_lizard_totals(lizard_totals)
        complexity = max(radon_score, lizard_score)

        # check testabilty (CI/CD configs)
        testability = 0.0
        for ci in [".github", ".gitlab-ci.yml", "azure-pipelines.yml"]:
            if os.path.exists(os.path.join(temp_dir, ci)):
                testability = 1.0
                break

        # check portability (check enviornment files)
        portability = 0.0
        if os.path.exists(os.path.join(temp_dir, "Dockerfile")):
            portability += 0.5
        if os.path.exists(os.path.join(temp_dir, "requirements.txt")) or os.path.exists(
            os.path.join(temp_dir, "environment.yml")
        ):
            portability += 0.5

        # reusability (check for README and docstrings)
        readme_path = os.path.join(temp_dir, "README.md")
        if os.path.exists(readme_path):
            readme_bonus = 0.5
        else:
            readme_bonus = 0

        reusability = max(docstring_ratio(temp_dir), readme_bonus)

        # compute weighted score
        final_score = (
            complexity * 0.70
            + reliability * 0.05
            + testability * 0.05
            + portability * 0.1
            + reusability * 0.1
        )

        return min(1.0, final_score)

    # remove the local copy of the repo
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def get_code_quality(url: str, url_type: str) -> Tuple[float, int]:
    """
    Function to get code quality if URL is a GitHub link.
    """

    start_time = time.time()
    score = 0.0

    if url_type == "code":
        # check code quality by downloading repo
        score = _check_code_repo_quality(url)

    latency = int((time.time() - start_time) * 1000)
    return score, latency
