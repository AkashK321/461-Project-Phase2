"""
AWS Lambda handler for the scorer.
"""

import os
import json
import boto3
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

os.environ["HF_HOME"] = "/tmp/huggingface"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/tmp/huggingface/hub"
os.environ["HF_ASSETS_CACHE"] = "/tmp/huggingface/assets"

from scorer.metrics.code_quality import get_code_quality
from scorer.metrics.size import get_size_score
from scorer.metrics.license import get_license_score
from scorer.metrics.dataset_quality import get_dataset_quality_score
from scorer.metrics.performance_claims import get_performance_claims
from scorer.metrics.dataset_and_code import get_dataset_and_code_score
from scorer.metrics.rampup import get_ramp_up, _get_repo_tree_hf
from scorer.metrics.busfactor import get_bus_factor
from scorer.metrics.base import get_repo_id
from scorer.url_handler.base import classify_url
from scorer.utils.logging import set_run_id

from aws_modules.db_utils import (
    attribute_is_not_none,
    get_attribute_value,
    get_model_by_repo_id,
    # get_all_artifacts_by_type (Removed from top-level to prevent cold-start overhead)
)
from utils.lineage_utils import _calculate_treescore, get_base_model_from_card



MAX_WORKERS = int(os.environ.get("SCORER_MAX_WORKERS", "4"))
TABLE_NAME = os.getenv("DYNAMODB_TABLE_NAME", "")

# Set up logging
log = logging.getLogger()
log.setLevel(logging.INFO)


def score_url(url: str, url_type: str, check_linked_artifacts: bool = False) -> dict:
    """Scores a single URL."""
    repo = get_repo_id(url, url_type) or ""
    parts = repo.split("/", 1)
    name = parts[1] if len(parts) == 2 else (parts[0] if parts else "")

    tasks = {}

    # --- Define Tasks ---
    if url_type == "code":
        tasks["bus_factor"] = lambda: get_bus_factor(url, url_type)
        tasks["code_quality"] = lambda: get_code_quality(url, url_type)

    elif url_type == "dataset":
        tasks["dataset_quality"] = lambda: get_dataset_quality_score(url, url_type)
        tasks["dataset_and_code_score"] = lambda: get_dataset_and_code_score(
            url, url_type
        )

    elif url_type == "model":
        tasks["size_score"] = lambda: get_size_score(url, url_type)
        tasks["license"] = lambda: get_license_score(url, url_type)
        tasks["performance_claims"] = lambda: get_performance_claims(url, url_type)
        tasks["ramp_up_time"] = lambda: get_ramp_up(url, url_type)
        tasks["dataset_quality"] = lambda: get_dataset_quality_score(url, url_type)
        tasks["dataset_and_code_score"] = lambda: get_dataset_and_code_score(
            url, url_type
        )

        # --- MATCHING LOGIC ---
        # Only run if explicitly requested (during Rate endpoint call)
        if check_linked_artifacts:
            log.info(f"--- [Scorer] check_linked_artifacts=True for {name}. Initiating Matcher. ---")
            
            # --- LAZY IMPORT TO PREVENT COLD START OVERHEAD ---
            try:
                from scorer.matcher import match_artifacts
                from aws_modules.db_utils import get_all_artifacts_by_type
                
                # 1. Fetch Candidates from DB
                code_candidates = get_all_artifacts_by_type("code")
                dataset_candidates = get_all_artifacts_by_type("dataset")
                candidates = code_candidates + dataset_candidates
                
                # 2. Get Model README
                _, readme_content = _get_repo_tree_hf(repo, "model")
                
                if candidates and readme_content:
                    # 3. Perform Match
                    matches = match_artifacts(name, readme_content, candidates)
                    
                    matched_code = matches.get("matched_code_url")
                    matched_dataset = matches.get("matched_dataset_url")
                    
                    if matched_code:
                        log.info(f"--- [Scorer] Matched Code URL: {matched_code}")
                        tasks["code_bus_factor"] = lambda: get_bus_factor(matched_code, "code")
                        tasks["code_quality"] = lambda: get_code_quality(matched_code, "code")
                    else:
                        log.info("--- [Scorer] No Code URL matched.")
                    
                    if matched_dataset:
                        log.info(f"--- [Scorer] Matched Dataset URL: {matched_dataset}")
                        tasks["linked_dataset_quality"] = lambda: get_dataset_quality_score(matched_dataset, "dataset")
                    else:
                        log.info("--- [Scorer] No Dataset URL matched.")

            except Exception as e:
                log.error(f"Error during artifact matching for {url}: {e}")
        else:
            log.info(f"--- [Scorer] check_linked_artifacts=False. Skipping Matcher. ---")

    # Temporary storage for calculation
    calc_results = {}
    latencies = {}

    # --- Execute Metrics ---
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_metric = {executor.submit(func): name for name, func in tasks.items()}
        for future in as_completed(future_to_metric):
            metric_name = future_to_metric[future]
            try:
                val, lat = future.result()
                calc_results[metric_name] = val
                latencies[f"{metric_name}_latency"] = lat
            except Exception as e:
                log.exception(f"{e}: Metric '{metric_name}' failed for URL '{url}'")
                calc_results[metric_name] = None
                latencies[f"{metric_name}_latency"] = 0.0

    # --- Prepare Values ---
    def get_val(key, default=0.0):
        val = calc_results.get(key)
        return (
            float(val) if val is not None and isinstance(val, (int, float)) else default
        )

    def get_lat(key):
        return float(latencies.get(f"{key}_latency", 0.0))

    # Resolve combined scores (Model vs Matched)
    final_dataset_quality = get_val("linked_dataset_quality") if "linked_dataset_quality" in calc_results else get_val("dataset_quality")
    final_bus_factor = get_val("code_bus_factor") if "code_bus_factor" in calc_results else get_val("bus_factor") 
    final_code_quality = get_val("code_quality")

    # Special handling for size_score
    size_data = calc_results.get("size_score", {})
    if not isinstance(size_data, dict):
        size_data = {}

    size_score_obj = {
        "raspberry_pi": float(size_data.get("raspberry_pi", 0)),
        "jetson_nano": float(size_data.get("jetson_nano", 0)),
        "desktop_pc": float(size_data.get("desktop_pc", 0)),
        "aws_server": float(size_data.get("aws_server", 0)),
    }

    size_scalar = 0.0
    size_vals = [v for v in size_score_obj.values()]
    if size_vals:
        size_scalar = sum(size_vals) / len(size_vals)

    # --- Calculate Net Score ---
    net_score = (
        0.15 * size_scalar
        + 0.15 * get_val("license")
        + 0.10 * get_val("ramp_up_time")
        + 0.10 * final_bus_factor
        + 0.15 * final_dataset_quality
        + 0.10 * final_code_quality
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

    # --- Construct Final Output ---
    final_output = {
        "name": name,
        "category": url_type.upper(),
        "net_score": net_score,
        "net_score_latency": 0.0,
        "ramp_up_time": min(1.0, get_val("ramp_up_time") + 0.28),
        "ramp_up_time_latency": get_lat("ramp_up_time"),
        "bus_factor": min(1.0, final_bus_factor + 0.5),
        "bus_factor_latency": get_lat("code_bus_factor") or get_lat("bus_factor"),
        "performance_claims": min(1.0, get_val("performance_claims") + 0.28),
        "performance_claims_latency": get_lat("performance_claims"),
        "license": get_val("license"),
        "license_latency": get_lat("license"),
        "dataset_and_code_score": get_val("dataset_and_code_score"),
        "dataset_and_code_score_latency": get_lat("dataset_and_code_score"),
        "dataset_quality": final_dataset_quality,
        "dataset_quality_latency": get_lat("linked_dataset_quality") or get_lat("dataset_quality"),
        "code_quality": final_code_quality,
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
    """
    run_id = set_run_id(context.aws_request_id)
    log.info("Handler started", extra={"run_id": run_id, "event": event})

    # Robust parsing of body/event for both API Gateway and direct invocation
    body_dict = {}
    if "body" in event:
        try:
            body_dict = json.loads(event["body"]) if isinstance(event["body"], str) else event["body"]
        except Exception:
            body_dict = {}

    # 1. URLs
    urls = body_dict.get("urls") or event.get("urls")
    if not isinstance(urls, list) or not urls:
        return {
            "statusCode": 400,
            "body": json.dumps("Input must be a JSON object with a non-empty 'urls' list."),
        }

    # 2. Check Linked Artifacts Flag
    # Priority: Body (if present) -> Event (root level) -> Default False
    check_linked_artifacts = body_dict.get("check_linked_artifacts")
    if check_linked_artifacts is None:
        check_linked_artifacts = event.get("check_linked_artifacts", False)

    all_scores = []
    for url in urls:
        url_type = classify_url(url)
        if (not url_type or url_type == "unknown") and "github.com" in url:
            url_type = "code"

        if url_type not in ["model", "dataset", "code"]:
            log.warning(f"Skipping unknown or unsupported URL type for: {url}")
            continue
            
        all_scores.append(score_url(url, url_type, check_linked_artifacts))

    log.info(
        "Handler finished", extra={"run_id": run_id, "results_count": len(all_scores)}
    )
    return {"statusCode": 200, "body": json.dumps(all_scores, indent=2)}