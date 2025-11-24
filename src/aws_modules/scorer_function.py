"""
AWS Lambda handler for the scorer.

This function is designed to be the entry point for an AWS Lambda function.
It receives a list of URLs via an event payload, processes them using the
existing scorer logic, and returns the results as a list of JSON objects.
"""

import traceback
import os

os.environ["HF_HOME"] = "/tmp/huggingface"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/tmp/huggingface/hub"
os.environ["HF_ASSETS_CACHE"] = "/tmp/huggingface/assets"
import json
import boto3
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from scorer.utils.logging import set_run_id
from aws_modules.db_utils import (
    attribute_is_not_none,
    get_attribute_value,
    get_model_by_repo_id,
)
from scorer.metrics.size import get_size_score
from scorer.metrics.license import get_license_score
from scorer.metrics.dataset_quality import get_dataset_quality_score

# from scorer.metrics.code_quality import get_code_quality
# from scorer.metrics.performance_claims import get_performance_claims
from scorer.metrics.dataset_and_code import get_dataset_and_code_score

# from scorer.metrics.rampup import get_ramp_up
# from scorer.metrics.busfactor import get_bus_factor
from scorer.metrics.base import get_repo_id
from scorer.url_handler.base import classify_url
from utils.lineage_utils import _calculate_treescore, get_base_model_from_card

# Import S3 test function
# from aws_modules.s3_utils import test_s3_operations

MAX_WORKERS = int(os.environ.get("SCORER_MAX_WORKERS", "4"))
TABLE_NAME = os.getenv("DYNAMODB_TABLE_NAME", "")
BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "")
USER_TABLE_NAME = os.getenv("USER_DYNAMODB_TABLE_NAME", "")
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "a-very-unsafe-default-secret")
SCORER_FUNCTION_NAME = os.getenv("SCORER_FUNCTION_NAME", "scorer_function")

dynamodb = boto3.resource("dynamodb")

# Set up logging
log = logging.getLogger()
log.setLevel(logging.INFO)


def score_url(url: str, url_type: str) -> dict:
    """Scores a single URL and returns a dictionary of results."""
    repo = get_repo_id(url, url_type) or ""
    parts = repo.split("/", 1)
    name = parts[1] if len(parts) == 2 else (parts[0] if parts else "")

    # TODO refactor metrics that rely on git
    # (will do this once metrics work for the phase 1 autograder)
    tasks = {}
    if url_type == "code":
        pass
        # tasks["code_quality"] = lambda: get_code_quality(url, url_type)
        # tasks["bus_factor"] = lambda: get_bus_factor(url, url_type)
        # tasks["ramp_up"] = lambda: get_ramp_up(url, url_type)
    elif url_type == "dataset":
        pass
        tasks["dataset_quality"] = lambda: get_dataset_quality_score(url, url_type)
        tasks["dataset_and_code_score"] = lambda: get_dataset_and_code_score(
            url, url_type
        )
    elif url_type == "model":
        pass
        tasks["size"] = lambda: get_size_score(url, url_type)
        tasks["license"] = lambda: get_license_score(url, url_type)
        # tasks["performance_claims"] = lambda: get_performance_claims(url, url_type)
        # tasks["bus_factor"] = lambda: get_bus_factor(url, url_type)
        # tasks["ramp_up"] = lambda: get_ramp_up(url, url_type)

    results = {"name": name, "category": url_type.upper()}
    latencies = {}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_metric = {executor.submit(func): name for name, func in tasks.items()}
        for future in as_completed(future_to_metric):
            metric_name = future_to_metric[future]
            try:
                val, lat = future.result()
                results[metric_name] = val
                latencies[f"{metric_name}_latency"] = lat
            except Exception as e:
                log.exception(f"{e}: Metric '{metric_name}' failed for URL '{url}'")
                results[metric_name] = 0.0 if metric_name not in ["size"] else {}
                latencies[f"{metric_name}_latency"] = 0

    # Calculate net_score (simplified for clarity, can be adjusted)
    # This logic should mirror your CLI's net score calculation
    size_score = 0.0
    if "size" in results and results["size"]:
        size_score = sum(results["size"].values()) / len(results["size"])

    net_score = (
        0.15 * size_score
        + 0.15 * results.get("license", 0.0)
        + 0.10 * results.get("ramp_up", 0.0)
        + 0.10 * results.get("bus_factor", 0.0)
        + 0.15 * results.get("dataset_quality", 0.0)
        + 0.10 * results.get("code_quality", 0.0)
        + 0.15 * results.get("performance_claims", 0.0)
        + 0.10 * results.get("dataset_and_code_score", 0.0)
    )
    results["net_score"] = round(net_score, 2)

    # --- Calculate Treescore if applicable ---
    # This model might already be in the DB if it's being re-scored.
    # If so, check if it has a parent model and calculate its treescore.
    try:
        item_in_db = get_model_by_repo_id(repo)
        log.info(f"Fetched item from DB for repo '{repo}': {item_in_db}")
        if item_in_db:
            item_id = item_in_db.get("id")
            if item_id and attribute_is_not_none(item_id, "base_model_repo_id"):
                log.info("Calculating treescore...")
                base_model_repo_id = get_attribute_value(item_id, "base_model_repo_id")
                log.info(f"Base model repo ID: {base_model_repo_id}")
                if base_model_repo_id:
                    treescore = _calculate_treescore(base_model_repo_id)
                    results["treescore"] = round(treescore, 2)
                    log.info(f"Treescore for {repo}: {treescore}")
        else:
            base_model_repo_id, lineage_type, source = get_base_model_from_card(repo)
            log.info(f"Base model from card: {base_model_repo_id}")
            if base_model_repo_id:
                log.info("Calculating treescore for new model...")
                treescore = _calculate_treescore(base_model_repo_id)
                results["treescore"] = round(treescore, 2)

    except Exception as e:
        log.error(f"Failed to calculate treescore for {repo}: {e}")
        log.error(traceback.format_exc())

    log.info(f"Scoring complete for URL: {url} | Results: {results}")

    # Combine results and latencies
    final_output = {**results, **latencies}
    return final_output

    
def handler(event, context):
    """
    AWS Lambda entry point.

    :param event: A dictionary containing the input data. Expected format:
                  `{"urls": ["url1", "url2", ...]}`
    :param context: A Lambda context object.
    :return: A dictionary with a status code and a body containing the scoring results.
    """
    body_str = event.get("body", "{}")
    body_dict = json.loads(body_str)

    urls = body_dict.get("urls")

    run_id = set_run_id(context.aws_request_id)
    log.info("Handler started", extra={"run_id": run_id, "event": event})

    # Check if this is an S3 test request
    # if event.get("test_s3", False):
    #     log.info("Running S3 test", extra={"run_id": run_id})
    #     test_result = test_s3_operations()
    #     return {"statusCode": 200, "body": json.dumps(test_result, indent=2)}

    urls = event.get("urls", [])
    if not isinstance(urls, list) or not urls:
        return {
            "statusCode": 400,
            "body": json.dumps(
                "Input must be a JSON object with a non-empty 'urls' list."
            ),
        }

    all_scores = []
    for url in urls:
        url_type = classify_url(url)
        if url_type not in ["model", "dataset", "code"]:
            log.warning(f"Skipping unknown or unsupported URL type for: {url}")
            continue
        all_scores.append(score_url(url, url_type))

    log.info(
        "Handler finished", extra={"run_id": run_id, "results_count": len(all_scores)}
    )
    return {"statusCode": 200, "body": json.dumps(all_scores, indent=2)}
