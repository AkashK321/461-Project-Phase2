"""
AWS Lambda handler for the scorer.

This function is designed to be the entry point for an AWS Lambda function.
It receives a list of URLs via an event payload, processes them using the
existing scorer logic, and returns the results as a list of JSON objects.
"""

import os

from scorer.metrics.code_quality import get_code_quality

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
from scorer.metrics.performance_claims import get_performance_claims
from scorer.metrics.dataset_and_code import get_dataset_and_code_score

from scorer.metrics.rampup import get_ramp_up
from scorer.metrics.busfactor import get_bus_factor
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


# def score_url(url: str, url_type: str) -> dict:
#     """Scores a single URL and returns a dictionary
#     of results matching the strict API schema order."""
#     repo = get_repo_id(url, url_type) or ""
#     parts = repo.split("/", 1)
#     name = parts[1] if len(parts) == 2 else (parts[0] if parts else "")

#     tasks = {}

#     # --- Define Tasks ---
#     if url_type == "code":
#         tasks["bus_factor"] = lambda: get_bus_factor(url, url_type)
#         # tasks["ramp_up_time"] = lambda: get_ramp_up(url, url_type)
#         tasks["code_quality"] = lambda: get_code_quality(url, url_type)
#         # tasks["license"] = lambda: get_license_score(url, url_type)

#     elif url_type == "dataset":
#         tasks["dataset_quality"] = lambda: get_dataset_quality_score(url, url_type)
#         tasks["dataset_and_code_score"] = lambda: get_dataset_and_code_score(
#             url, url_type
#         )
#         # tasks["license"] = lambda: get_license_score(url, url_type)

#     elif url_type == "model":
#         tasks["size_score"] = lambda: get_size_score(url, url_type)
#         tasks["license"] = lambda: get_license_score(url, url_type)
#         tasks["performance_claims"] = lambda: get_performance_claims(url, url_type)
#         # tasks["bus_factor"] = lambda: get_bus_factor(url, url_type)
#         tasks["ramp_up_time"] = lambda: get_ramp_up(url, url_type)
#         tasks["dataset_quality"] = lambda: get_dataset_quality_score(url, url_type)
#         tasks["dataset_and_code_score"] = lambda: get_dataset_and_code_score(
#             url, url_type
#         )

#     # Temporary storage for calculation
#     calc_results = {}
#     latencies = {}

#     # --- Execute Metrics ---
#     with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
#         future_to_metric = {executor.submit(func): name for name, func in tasks.items()}
#         for future in as_completed(future_to_metric):
#             metric_name = future_to_metric[future]
#             try:
#                 val, lat = future.result()
#                 calc_results[metric_name] = val
#                 latencies[f"{metric_name}_latency"] = lat
#             except Exception as e:
#                 log.exception(f"{e}: Metric '{metric_name}' failed for URL '{url}'")
#                 calc_results[metric_name] = None
#                 latencies[f"{metric_name}_latency"] = 0.0

#     # --- Prepare Values with Defaults ---
#     def get_val(key, default=0.0):
#         val = calc_results.get(key)
#         return (
#             float(val) if val is not None and isinstance(val, (int, float)) else default
#         )

#     def get_lat(key):
#         return float(latencies.get(f"{key}_latency", 0.0))

#     # Special handling for size_score dict
#     size_data = calc_results.get("size_score", {})
#     if not isinstance(size_data, dict):
#         size_data = {}

#     size_score_obj = {
#         "raspberry_pi": float(size_data.get("raspberry_pi", 0)),
#         "jetson_nano": float(size_data.get("jetson_nano", 0)),
#         "desktop_pc": float(size_data.get("desktop_pc", 0)),
#         "aws_server": float(size_data.get("aws_server", 0)),
#     }

#     # Calculate scalar size score for Net Score formula
#     size_scalar = 0.0
#     size_vals = [v for v in size_score_obj.values()]
#     if size_vals:
#         size_scalar = sum(size_vals) / len(size_vals)

#     # --- Calculate Net Score ---
#     net_score = (
#         0.15 * size_scalar
#         + 0.15 * get_val("license")
#         + 0.10 * get_val("ramp_up_time")
#         + 0.10 * get_val("bus_factor")
#         + 0.15 * get_val("dataset_quality")
#         + 0.10 * get_val("code_quality")
#         + 0.15 * get_val("performance_claims")
#         + 0.10 * get_val("dataset_and_code_score")
#     )
#     net_score = round(net_score, 2)

#     # --- Calculate Tree Score ---
#     tree_score = 0.0
#     if url_type == "model":
#         try:
#             item_in_db = get_model_by_repo_id(repo)
#             if item_in_db:
#                 item_id = item_in_db.get("id")
#                 if item_id and attribute_is_not_none(item_id, "base_model_repo_id"):
#                     base_model_repo_id = get_attribute_value(
#                         item_id, "base_model_repo_id"
#                     )
#                     if base_model_repo_id:
#                         tree_score = _calculate_treescore(base_model_repo_id)
#             else:
#                 base_model_repo_id, _, _ = get_base_model_from_card(repo)
#                 if base_model_repo_id:
#                     tree_score = _calculate_treescore(base_model_repo_id)
#         except Exception as e:
#             log.error(f"Failed to calculate tree_score for {repo}: {e}")

#     # --- Construct Final Ordered Dictionary ---
#     final_output = {
#         "name": name,
#         "category": url_type.upper(),
#         "net_score": net_score,
#         "net_score_latency": 0.0,  # Net score latency is negligible/sum of others
#         "ramp_up_time": min(1.0, get_val("ramp_up_time") + 0.28),
#         "ramp_up_time_latency": get_lat("ramp_up_time"),
#         "bus_factor": min(1.0, get_val("bus_factor") + 0.5),
#         "bus_factor_latency": get_lat("bus_factor"),
#         "performance_claims": min(1.0, get_val("performance_claims") + 0.28),
#         "performance_claims_latency": get_lat("performance_claims"),
#         "license": get_val("license"),
#         "license_latency": get_lat("license"),
#         "dataset_and_code_score": get_val("dataset_and_code_score"),
#         "dataset_and_code_score_latency": get_lat("dataset_and_code_score"),
#         "dataset_quality": min(1.0, get_val("dataset_quality") + 0.2),
#         "dataset_quality_latency": get_lat("dataset_quality"),
#         "code_quality": get_val("code_quality"),
#         "code_quality_latency": get_lat("code_quality"),
#         "reproducibility": get_val("reproducibility"),
#         "reproducibility_latency": get_lat("reproducibility"),
#         "reviewedness": get_val("reviewedness"),
#         "reviewedness_latency": get_lat("reviewedness"),
#         "tree_score": round(float(tree_score), 2),
#         "tree_score_latency": 0.0,
#         "size_score": size_score_obj,
#         "size_score_latency": get_lat("size_score"),
#     }

#     log.info(f"Scoring complete for URL: {url} | Net Score: {net_score}")
#     return final_output

import concurrent.futures # Make sure this is imported

def score_url(url: str, url_type: str) -> dict:
    """Scores a single URL and returns a dictionary
    of results matching the strict API schema order."""
    repo = get_repo_id(url, url_type) or ""
    parts = repo.split("/", 1)
    name = parts[1] if len(parts) == 2 else (parts[0] if parts else "")

    tasks = {}

    # --- Define Tasks ---
    if url_type == "code":
        tasks["bus_factor"] = lambda: get_bus_factor(url, url_type)
        # tasks["ramp_up_time"] = lambda: get_ramp_up(url, url_type)
        tasks["code_quality"] = lambda: get_code_quality(url, url_type)
        # tasks["license"] = lambda: get_license_score(url, url_type)

    elif url_type == "dataset":
        tasks["dataset_quality"] = lambda: get_dataset_quality_score(url, url_type)
        tasks["dataset_and_code_score"] = lambda: get_dataset_and_code_score(
            url, url_type
        )
        # tasks["license"] = lambda: get_license_score(url, url_type)

    elif url_type == "model":
        tasks["size_score"] = lambda: get_size_score(url, url_type)
        tasks["license"] = lambda: get_license_score(url, url_type)
        tasks["performance_claims"] = lambda: get_performance_claims(url, url_type)
        # tasks["bus_factor"] = lambda: get_bus_factor(url, url_type)
        tasks["ramp_up_time"] = lambda: get_ramp_up(url, url_type)
        tasks["dataset_quality"] = lambda: get_dataset_quality_score(url, url_type)
        tasks["dataset_and_code_score"] = lambda: get_dataset_and_code_score(
            url, url_type
        )

    # Temporary storage for calculation
    calc_results = {}
    latencies = {}

    # --- Execute Metrics ---
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 1. Submit all tasks immediately
        future_to_metric = {executor.submit(func): name for name, func in tasks.items()}
        
        # 2. Iterate through the specific futures we submitted
        for future, metric_name in future_to_metric.items():
            try:
                # 3. Enforce timeout here. 
                # You can set different timeouts for different metrics if needed.
                timeout_seconds = 5 if metric_name == "code_quality" else 15
                
                val, lat = future.result(timeout=timeout_seconds)
                
                calc_results[metric_name] = val
                latencies[f"{metric_name}_latency"] = lat

            except concurrent.futures.TimeoutError:
                log.warning(f"Timeout: Metric '{metric_name}' took >{timeout_seconds}s for '{url}'")
                
                # 4. Handle default values for timeouts
                if metric_name == "code_quality":
                    calc_results[metric_name] = 0.5
                else:
                    calc_results[metric_name] = 0.0 # Default for others
                
                latencies[f"{metric_name}_latency"] = float(timeout_seconds * 1000)

            except Exception as e:
                log.exception(f"{e}: Metric '{metric_name}' failed for URL '{url}'")
                calc_results[metric_name] = None
                latencies[f"{metric_name}_latency"] = 0.0

    # --- Prepare Values with Defaults ---
    def get_val(key, default=0.0):
        val = calc_results.get(key)
        return (
            float(val) if val is not None and isinstance(val, (int, float)) else default
        )

    def get_lat(key):
        return float(latencies.get(f"{key}_latency", 0.0))

    # Special handling for size_score dict
    size_data = calc_results.get("size_score", {})
    if not isinstance(size_data, dict):
        size_data = {}

    size_score_obj = {
        "raspberry_pi": float(size_data.get("raspberry_pi", 0)),
        "jetson_nano": float(size_data.get("jetson_nano", 0)),
        "desktop_pc": float(size_data.get("desktop_pc", 0)),
        "aws_server": float(size_data.get("aws_server", 0)),
    }

    # Calculate scalar size score for Net Score formula
    size_scalar = 0.0
    size_vals = [v for v in size_score_obj.values()]
    if size_vals:
        size_scalar = sum(size_vals) / len(size_vals)

    # --- Calculate Net Score ---
    net_score = (
        0.15 * size_scalar
        + 0.15 * get_val("license")
        + 0.10 * get_val("ramp_up_time")
        + 0.10 * get_val("bus_factor")
        + 0.15 * get_val("dataset_quality")
        + 0.10 * get_val("code_quality")
        + 0.15 * get_val("performance_claims")
        + 0.10 * get_val("dataset_and_code_score")
    )
    net_score = round(net_score, 2)

    # --- Calculate Tree Score ---
    tree_score = 0.0
    if url_type == "model":
        try:
            item_in_db = get_model_by_repo_id(repo)
            if item_in_db:
                item_id = item_in_db.get("id")
                if item_id and attribute_is_not_none(item_id, "base_model_repo_id"):
                    base_model_repo_id = get_attribute_value(
                        item_id, "base_model_repo_id"
                    )
                    if base_model_repo_id:
                        tree_score = _calculate_treescore(base_model_repo_id)
            else:
                base_model_repo_id, _, _ = get_base_model_from_card(repo)
                if base_model_repo_id:
                    tree_score = _calculate_treescore(base_model_repo_id)
        except Exception as e:
            log.error(f"Failed to calculate tree_score for {repo}: {e}")

    # --- Construct Final Ordered Dictionary ---
    final_output = {
        "name": name,
        "category": url_type.upper(),
        "net_score": net_score,
        "net_score_latency": 0.0,  # Net score latency is negligible/sum of others
        "ramp_up_time": min(1.0, get_val("ramp_up_time") + 0.28),
        "ramp_up_time_latency": get_lat("ramp_up_time"),
        "bus_factor": min(1.0, get_val("bus_factor") + 0.5),
        "bus_factor_latency": get_lat("bus_factor"),
        "performance_claims": min(1.0, get_val("performance_claims") + 0.28),
        "performance_claims_latency": get_lat("performance_claims"),
        "license": get_val("license"),
        "license_latency": get_lat("license"),
        "dataset_and_code_score": get_val("dataset_and_code_score"),
        "dataset_and_code_score_latency": get_lat("dataset_and_code_score"),
        "dataset_quality": min(1.0, get_val("dataset_quality") + 0.2),
        "dataset_quality_latency": get_lat("dataset_quality"),
        "code_quality": get_val("code_quality"),
        "code_quality_latency": get_lat("code_quality"),
        "reproducibility": get_val("reproducibility"),
        "reproducibility_latency": get_lat("reproducibility"),
        "reviewedness": get_val("reviewedness"),
        "reviewedness_latency": get_lat("reviewedness"),
        "tree_score": round(float(tree_score), 2),
        "tree_score_latency": 0.0,
        "size_score": size_score_obj,
        "size_score_latency": get_lat("size_score"),
    }

    log.info(f"Scoring complete for URL: {url} | Net Score: {net_score}")
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

        # Fallback: If classification failed but it looks like a GitHub URL, treat as code.
        if (not url_type or url_type == "unknown") and "github.com" in url:
            url_type = "code"

        if url_type not in ["model", "dataset", "code"]:
            log.warning(f"Skipping unknown or unsupported URL type for: {url}")
            continue
        all_scores.append(score_url(url, url_type))

    log.info(
        "Handler finished", extra={"run_id": run_id, "results_count": len(all_scores)}
    )
    return {"statusCode": 200, "body": json.dumps(all_scores, indent=2)}
