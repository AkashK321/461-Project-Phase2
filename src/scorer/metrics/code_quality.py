"""
Evaluates the quality of the code with strict limits to prevent timeouts.
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

# --- STRICT FILTERS ---
EXCLUDED_EXTS = {
    # Images/Media
    '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.svg', '.webp',
    '.mp4', '.mov', '.avi', '.mp3', '.wav',
    # Archives
    '.zip', '.tar', '.gz', '.7z', '.rar',
    # Documents
    '.pdf', '.doc', '.docx', '.ppt', '.pptx', '.xls', '.xlsx', '.txt', '.md',
    # Binary/Compiled
    '.pyc', '.pyo', '.pyd', '.o', '.obj', '.dll', '.exe', '.so', '.dylib', 
    '.class', '.jar', '.bin', '.onnx', '.pb', '.h5', '.hdf5', '.safetensors',
    # Git
    '.pack', '.idx', '.sample'
}

EXCLUDED_DIRS = {
    '__pycache__', '.git', '.github', '.idea', '.vscode', '.venv', 'venv', 'env', 'node_modules',
    'bin', 'obj', 'build', 'dist', 'target',
    # Test directories (Crucial for repos like microsoft/git)
    't', 'test', 'tests', 'testing', 'spec', 'specs',
    # Docs & Translations
    'doc', 'docs', 'documentation', 'po', 'site', 'examples', 'samples', 'contrib'
}

def is_allowed(filename: str) -> bool:
    """Check if a file should be extracted."""
    parts = filename.split('/')
    
    # Block hidden files/dirs
    if any(p.startswith('.') for p in parts):
        return False

    # Block specific dirs
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
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=5)
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
            # STRICT 5s TIMEOUT for analysis
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=5)
            break
        except Exception:
            result = None
            continue
            
    if not result or result.returncode != 0:
        return None

    # Parse Lizard Output (Total nloc ...)
    total_row = None
    for line in result.stdout.splitlines():
        # Look for the summary line which usually starts with NLOC count
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
    if not totals: return 0.0
    
    avg_ccn = totals.get("Avg CCN", 0)
    if avg_ccn <= 5: ccn_score = 1.0
    elif avg_ccn <= 10: ccn_score = 0.8
    elif avg_ccn <= 20: ccn_score = 0.5
    else: ccn_score = 0.2

    avg_nloc = totals.get("Avg NLOC", 0)
    if avg_nloc <= 30: nloc_score = 1.0
    elif avg_nloc <= 40: nloc_score = 0.8
    elif avg_nloc <= 100: nloc_score = 0.5
    else: nloc_score = 0.2

    return (ccn_score + nloc_score) / 2.0

def docstring_ratio(path: str) -> float:
    total = 0
    documented = 0
    # Limit recursion depth or file count if needed, but os.walk is usually fast enough on filtered dirs
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith(".py"):
                try:
                    with open(os.path.join(root, file), "r", encoding="utf-8", errors="ignore") as fh:
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
    
    # --- LIMITS ---
    MAX_FILES = 100      # Only extract first 100 valid source files
    MAX_TIME = 5.0       # Only spend 5 seconds extracting
    # --------------

    try:
        # 1. Resolve Repo Path
        if "github.com" in code_url:
            repo_path = code_url.split("github.com/")[-1].strip("/")
            if repo_path.endswith(".git"): repo_path = repo_path[:-4]
        else:
            repo_path = code_url.split("/")[-1]

        # 2. Download Stream
        zip_url = f"https://github.com/{repo_path}/archive/refs/heads/main.zip"
        try:
            r = requests.get(zip_url, stream=True, timeout=8)
            if r.status_code != 200:
                zip_url = f"https://github.com/{repo_path}/archive/refs/heads/master.zip"
                r = requests.get(zip_url, stream=True, timeout=8)
        except Exception:
            return 0.0

        if not r or r.status_code != 200:
            return 0.0

        # 3. Extract with STRICT LIMITS
        start_extract = time.time()
        file_count = 0
        
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            for file_info in z.infolist():
                # Time Check
                if (time.time() - start_extract) > MAX_TIME:
                    break
                # Count Check
                if file_count >= MAX_FILES:
                    break
                
                # Filter Check
                if not file_info.filename.endswith('/') and is_allowed(file_info.filename):
                    z.extract(file_info, temp_dir)
                    file_count += 1

        # If we extracted nothing (maybe filter was too strict or download failed), return 0
        if file_count == 0:
            return 0.0

        # 4. Flatten Directory
        items = os.listdir(temp_dir)
        if len(items) == 1 and os.path.isdir(os.path.join(temp_dir, items[0])):
            top_level = os.path.join(temp_dir, items[0])
            for item in os.listdir(top_level):
                shutil.move(os.path.join(top_level, item), temp_dir)
            os.rmdir(top_level)

        # 5. Analysis (Run on the small subset of files)
        
        # Reliability (Simple grep)
        reliability = 0.0
        for root, _, files in os.walk(temp_dir):
            for f in files:
                if "test" in f.lower():
                    reliability = 0.7
                    break
            if reliability > 0: break
            
        # Complexity
        radon_score = run_radon(temp_dir)
        lizard_totals = run_lizard(temp_dir)
        lizard_score = score_from_lizard_totals(lizard_totals) if lizard_totals else 0.0
        complexity = max(radon_score, lizard_score)

        # Testability / Portability (Check existence of config files)
        # (We check these even if we didn't extract the full content, 
        # but since we filtered, we might miss them if they were in excluded dirs.
        # However, root-level configs like .github/ or Dockerfile are usually extracted first/early)
        testability = 0.5 # Assume some testability if we got this far
        portability = 0.0
        if os.path.exists(os.path.join(temp_dir, "Dockerfile")): portability = 1.0
        elif os.path.exists(os.path.join(temp_dir, "requirements.txt")): portability = 1.0

        reusability = max(docstring_ratio(temp_dir), 0.0)

        final_score = (
            complexity * 0.70
            + reliability * 0.05
            + testability * 0.05
            + portability * 0.1
            + reusability * 0.1
        )
        return min(1.0, max(0.0, final_score))

    except Exception as e:
        print(f"Code quality check failed: {e}")
        return 0.0
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

def get_code_quality(url: str, url_type: str) -> Tuple[float, int]:
    start = time.time()
    score = 0.0
    if url_type == "code":
        score = _check_code_repo_quality(url)
    return score, int((time.time() - start) * 1000)