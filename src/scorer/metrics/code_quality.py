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

# --- Constants for Filtering ---
EXCLUDED_EXTS = {
    # Images & Media
    '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.svg', '.webp',
    '.mp4', '.mov', '.avi', '.mp3', '.wav',
    # Archives
    '.zip', '.tar', '.gz', '.7z', '.rar',
    # Docs
    '.pdf', '.doc', '.docx', '.ppt', '.pptx', '.xls', '.xlsx',
    # Compiled / Binary / Models
    '.pyc', '.pyo', '.pyd', '.o', '.obj', '.dll', '.exe', '.so', '.dylib', '.class', '.jar',
    '.bin', '.onnx', '.pb', '.h5', '.hdf5', '.safetensors',
    # Other
    '.DS_Store'
}

EXCLUDED_DIRS = {
    '__pycache__', '.git', '.idea', '.vscode', '.venv', 'venv', 'env', 'node_modules',
    'bin', 'obj', 'build', 'dist', 'target'
}

def is_allowed(filename: str) -> bool:
    """Check if a file should be extracted based on extension and path."""
    parts = filename.split('/')
    
    # Check for excluded directories
    for part in parts:
        if part in EXCLUDED_DIRS:
            return False
            
    # Check extension
    _, ext = os.path.splitext(filename)
    if ext.lower() in EXCLUDED_EXTS:
        return False
        
    return True

def run_radon(path: str) -> float:
    """
    Function to run radon, a Python tool that analyzes source code complexity and
    maintainability.
    """
    # Use 'radon mi' which calculates Maintainability Index
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
            # If radon fails (e.g. no python files), return 0
            return 0.0

    if result is None:
        return 0.0

    score_map = {"A": 1.0, "B": 0.8, "C": 0.6, "D": 0.4, "E": 0.2, "F": 0.0}
    vals = []
    
    # Parse output: "path/to/file.py - A"
    for line in result.stdout.splitlines():
        if " - " in line:
            parts = line.split(" -")
            if len(parts) >= 2:
                grade = parts[-1].strip()[:1] # Get first letter
                vals.append(score_map.get(grade, 0.0))
                
    return sum(vals) / len(vals) if vals else 0.0


def run_lizard(path: str) -> Optional[Dict]:
    """
    Function to run lizard, a multi-language code analysis tool.
    """
    # Lizard can scan a directory recursively
    for cmd in ([sys.executable, "-m", "lizard", path], ["lizard", path]):
        try:
            # Set a timeout for the analysis tool itself too
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=20)
            break
        except (FileNotFoundError, subprocess.TimeoutExpired):
            result = None
            continue
        except subprocess.CalledProcessError:
            return None
            
    if result is None or result.returncode != 0:
        return None

    # find the last summary row using regex
    total_row = None
    for line in result.stdout.splitlines():
        # match a line that starts with NLOC ...
        if "Total nloc" in line:
             # usually lizard prints a header, but we want the numerical summary
             pass
        # standard lizard summary line starts with a number (NLOC)
        if re.match(r"^\s*\d+\s+\d+", line.strip()):
            total_row = line.strip()

    if not total_row:
        return None

    parts = total_row.split()
    # Lizard output format: NLOC Avg.NLOC Avg.CCN Avg.token function_cnt ...
    if len(parts) < 6:
        return None

    try:
        totals = {
            "Total NLOC": float(parts[0]),
            "Avg NLOC": float(parts[1]),
            "Avg CCN": float(parts[2]),
            "Avg Tokens": float(parts[3]),
            "Function Count": int(parts[4]),
            "Warning Count": int(parts[5]),
        }
        return totals
    except ValueError:
        return None


def score_from_lizard_totals(totals: dict) -> float:
    """
    Calculate final score from lizard metrics.
    """
    if not totals:
        return 0.0

    # Metric 1: Cyclomatic complexity (CCN)
    avg_ccn = totals.get("Avg CCN", 0)
    if avg_ccn <= 5: ccn_score = 1.0
    elif avg_ccn <= 10: ccn_score = 0.8
    elif avg_ccn <= 20: ccn_score = 0.5
    else: ccn_score = 0.2

    # Metric 2: Average NLOC (function size)
    avg_nloc = totals.get("Avg NLOC", 0)
    if avg_nloc <= 30: nloc_score = 1.0
    elif avg_nloc <= 40: nloc_score = 0.8
    elif avg_nloc <= 100: nloc_score = 0.5
    else: nloc_score = 0.2

    # Metric 3: Warnings
    warnings = totals.get("Warning Count", 0)
    if warnings == 0: warning_score = 1.0
    elif warnings <= 2: warning_score = 0.7
    elif warnings <= 5: warning_score = 0.4
    else: warning_score = 0.1

    weights = [0.5, 0.3, 0.2]
    components = [ccn_score, nloc_score, warning_score]
    final_score = sum(w * c for w, c in zip(weights, components)) / sum(weights)

    return final_score


def docstring_ratio(path: str) -> float:
    """
    Count docstring coverage in Python files.
    """
    total = 0
    documented = 0

    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as fh:
                        tree = ast.parse(fh.read())
                        for node in ast.walk(tree):
                            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                                total += 1
                                if ast.get_docstring(node):
                                    documented += 1
                except Exception:
                    continue

    if total == 0:
        return 0.0 # Neutral/Low if no python code found
    return documented / total


def _check_code_repo_quality(code_url: str) -> float:
    """
    Analyzes code quality by downloading a filtered zip of the repo.
    """
    temp_dir = tempfile.mkdtemp()
    try:
        # --- DOWNLOAD LOGIC ---
        try:
            # 1. Parse Owner/Repo
            if "github.com" in code_url:
                repo_path = code_url.split("github.com/")[-1].strip("/")
                if repo_path.endswith(".git"):
                    repo_path = repo_path[:-4]
            else:
                repo_path = code_url.split("/")[-1]

            # 2. Try 'main', then 'master'
            zip_url = f"https://github.com/{repo_path}/archive/refs/heads/main.zip"
            
            # Request with TIMEOUT
            try:
                r = requests.get(zip_url, stream=True, timeout=10)
            except requests.RequestException:
                r = None

            if not r or r.status_code != 200:
                zip_url = f"https://github.com/{repo_path}/archive/refs/heads/master.zip"
                try:
                    r = requests.get(zip_url, stream=True, timeout=10)
                except requests.RequestException:
                    pass

            if not r or r.status_code != 200:
                print(f"Failed to download zip from {code_url}")
                return 0.0

            # 3. Extract with FILTERING
            # Stream the response into BytesIO to avoid saving the huge zip to disk
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                for file_info in z.infolist():
                    # Only extract if it's not a folder AND it passes the filter
                    if not file_info.filename.endswith('/') and is_allowed(file_info.filename):
                        # Extract only this specific file
                        z.extract(file_info, temp_dir)

            # 4. Flatten Directory (if single top-level folder exists)
            items = os.listdir(temp_dir)
            if len(items) == 1 and os.path.isdir(os.path.join(temp_dir, items[0])):
                top_level = os.path.join(temp_dir, items[0])
                for item in os.listdir(top_level):
                    shutil.move(os.path.join(top_level, item), temp_dir)
                os.rmdir(top_level)

        except Exception as e:
            print(f"Error downloading/extracting {code_url}: {e}")
            return 0.0

        # --- ANALYSIS ---

        # 1. Reliability (Tests)
        reliability = 0.0
        has_tests = False
        frameworks = ["pytest", "unittest", "mocha", "jest", "junit", "testng"]
        
        # Fast walk for tests
        for root, dirs, files in os.walk(temp_dir):
            if "test" in root.lower(): 
                has_tests = True
            
            for file in files:
                f_lower = file.lower()
                if "test" in f_lower:
                    has_tests = True
                
                # Check for test config files/frameworks
                if any(fw in f_lower for fw in frameworks):
                    reliability = 1.0
                    break
            if reliability == 1.0: break
        
        if reliability < 1.0 and has_tests:
            reliability = 0.7

        # 2. Complexity (Radon/Lizard)
        # These now run ONLY on the filtered files, so they should be much faster.
        radon_score = run_radon(temp_dir)
        lizard_totals = run_lizard(temp_dir)
        lizard_score = 0.0
        if lizard_totals:
            lizard_score = score_from_lizard_totals(lizard_totals)
        
        # Take the better of the two or average? Using max as per original logic
        complexity = max(radon_score, lizard_score)

        # 3. Testability (CI Configs)
        testability = 0.0
        # Check specific locations for CI files
        ci_files = [
            ".github/workflows", 
            ".gitlab-ci.yml", 
            "azure-pipelines.yml", 
            ".travis.yml", 
            "circle.yml"
        ]
        
        for ci in ci_files:
            full_path = os.path.join(temp_dir, ci)
            if os.path.exists(full_path):
                testability = 1.0
                break

        # 4. Portability (Docker/Env)
        portability = 0.0
        if os.path.exists(os.path.join(temp_dir, "Dockerfile")) or \
           os.path.exists(os.path.join(temp_dir, "docker-compose.yml")):
            portability += 0.5
            
        if os.path.exists(os.path.join(temp_dir, "requirements.txt")) or \
           os.path.exists(os.path.join(temp_dir, "environment.yml")) or \
           os.path.exists(os.path.join(temp_dir, "package.json")) or \
           os.path.exists(os.path.join(temp_dir, "pom.xml")):
            portability += 0.5
        
        # Cap portability
        if portability > 1.0: portability = 1.0

        # 5. Reusability (Readme + Docstrings)
        reusability = 0.0
        # Check standard README names
        for name in ["README.md", "README", "readme.md", "readme.txt"]:
            if os.path.exists(os.path.join(temp_dir, name)):
                reusability += 0.5
                break
                
        doc_score = docstring_ratio(temp_dir)
        reusability = max(reusability, doc_score) # simplified bonus logic

        # Final Weighted Score
        final_score = (
            complexity * 0.70
            + reliability * 0.05
            + testability * 0.05
            + portability * 0.1
            + reusability * 0.1
        )

        return min(1.0, max(0.0, final_score))

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def get_code_quality(url: str, url_type: str) -> Tuple[float, int]:
    """
    Function to get code quality if URL is a GitHub link.
    """
    start_time = time.time()
    score = 0.0

    if url_type == "code":
        score = _check_code_repo_quality(url)

    latency = int((time.time() - start_time) * 1000)
    return score, latency