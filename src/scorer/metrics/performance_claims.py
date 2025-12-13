"""
Evaluate model performance claims.
"""

import os
import shutil
import tempfile
import time
from typing import Tuple
import requests
import zipfile
import io
from huggingface_hub import hf_hub_download
from dotenv import load_dotenv
from pathlib import Path
import re
import logging

logger = logging.getLogger(__name__)


def get_performance_claims(url: str, url_type: str) -> Tuple[float, int]:
    """
    Function to get model or code performance claims based on URL type.
    """

    start_time = time.time()
    score = 0.0
    logger.info(f"Evaluating performance claims for {url_type} URL: {url}")

    if url_type == "code":
        # clone GitHub repo and check readme for performance claims
        score = _check_code_repo_performance(url)
    elif url_type == "model":
        # check Hugging face model card for performance claims
        score = _check_model_card_performance(url)

    logger.info(f"Model performance claims score: {score}")

    latency = int((time.time() - start_time) * 1000)

    return score, latency


def _check_code_repo_performance(code_url: str) -> float:
    """
    Function to check the code repo for performance claims without using GitPython.
    Downloads the repo as a zip archive.
    """

    score = 0.0
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
            return 0.0

        # 3. Locate files (GitHub zips create a nested root folder)
        # We traverse the temp_dir to find content regardless of the root folder name
        readme_text = ""
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
                            readme_text = f.read().lower()
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

    return score


def _check_model_card_performance(model_url: str) -> float:
    """
    Function to check the model card/README on Hugging Face for performance claims.
    """

    load_dotenv(dotenv_path=Path(__file__).resolve().parents[3] / ".env")
    hf_token = os.getenv("HF_TOKEN")

    score = 0.0

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
            text = f.read().lower()

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

    return round(score, 2)


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
