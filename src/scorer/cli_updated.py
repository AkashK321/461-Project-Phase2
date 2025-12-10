"""
CLI for scoring tool.
"""

from __future__ import annotations
import sys
import io
import argparse
from typing import List
from pathlib import Path
import time
import os
from contextlib import redirect_stdout
import math
import json
import requests
from utils.logging import setup_logging, set_run_id, get_logger
from url_handler.base import classify_url

from metrics.size import get_size_score
from metrics.license import get_license_score
from metrics.dataset_quality import get_dataset_quality_score
from metrics.code_quality import get_code_quality
from metrics.performance_claims import get_performance_claims
from metrics.dataset_and_code import get_dataset_and_code_score
from metrics.rampup import get_ramp_up
from metrics.busfactor import get_bus_factor
from metrics.base import get_repo_id
from concurrent.futures import ThreadPoolExecutor, as_completed

MAX_WORKERS = int(os.environ.get("SCORER_MAX_WORKERS", "4"))
_BOOT_STDOUT = sys.stdout
sys.stdout = io.StringIO()

SIZE_WEIGHT = 0.18
LICENSE_WEIGHT = 0.14
RAMP_UP_WEIGHT = 0.12
BUS_FACTOR_WEIGHT = 0.05
DATASET_QUALITY_WEIGHT = 0.14
CODE_QUALITY_WEIGHT = 0.08
PERFORMANCE_CLAIMS_WEIGHT = 0.17
DATASET_AND_CODE_WEIGHT = 0.12


def safe_metric_value(val: float) -> float:
    """
    Normalize a metric value so that weird returns (None, NaN, inf, wrong type)
    don't blow up the net score calculation.
    """
    try:
        if val is None:
            return 0.0
        if isinstance(val, bool):
            val = float(val)
        if not isinstance(val, (int, float)):
            return 0.0
        if math.isnan(val) or math.isinf(val):
            return 0.0
        return float(val)
    except Exception:
        return 0.0


def compute_size_score(size_dict: dict) -> float:
    """
    Compute an average size score from the per-device dict.
    Be robust if the metric returns {}, None, or non-numeric values.
    """
    if not isinstance(size_dict, dict) or not size_dict:
        return 0.0
    vals = []
    for v in size_dict.values():
        if isinstance(v, (int, float)):
            if not (math.isnan(v) or math.isinf(v)):
                vals.append(float(v))
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


def safe_round(val: float, ndigits: int = 2) -> float:
    """
    Round a metric to ndigits, clamping into [0, 1] and handling bad values.
    For normal values in [0, 1] this behaves like round(val, ndigits).
    """
    try:
        v = safe_metric_value(val)
        # Clamp to [0, 1] because all metrics are defined in that range.
        if v < 0.0:
            v = 0.0
        elif v > 1.0:
            v = 1.0
        return round(v, ndigits)
    except Exception:
        return 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CLI for scoring models, datasets, and code."
    )
    parser.add_argument(
        "url_file",
        type=Path,
        help="Path to a newline-delimited file containing URLS of model/dataset/code",
    )
    parser.add_argument("--log-file", type=Path, default=None)
    parser.add_argument(
        "--log-level",
        type=int,
        choices=[0, 1, 2],
        default=int(os.environ.get("LOG_LEVEL", "0")),
    )
    parser.add_argument("--log-text", action="store_true")
    parser.add_argument("--run-id", default=None)
    return parser.parse_args()


def read_urls(file_path: Path) -> List[List[str]]:
    if not file_path.exists():
        raise FileNotFoundError(f"URL file {file_path} does not exist.")
    urls: List[List[str]] = []
    with file_path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = [u.strip() for u in line.split(",") if u.strip()]
            if parts:
                urls.append(parts)
    return urls


def main() -> None:
    args = parse_args()
    url_file_path = args.url_file.resolve()

    if args.log_file is not None:
        raw_log = str(args.log_file)

        # Reject empty, whitespace-only, or missing
        if not raw_log.strip():
            print("Error: invalid log file path", file=sys.stderr)
            sys.exit(1)

        # Use raw args.log_file unconditionally
        os.environ["LOG_FILE"] = raw_log

    else:
        # If no argument was passed, do NOT overwrite existing env var.
        os.environ.setdefault("LOG_FILE", "logs/scorer.log")

    log_file = os.environ.get("LOG_FILE", "")
    if not log_file.strip():
        print("Error: invalid log file path", file=sys.stderr)
        sys.exit(1)

    log_path = Path(log_file)
    parent = log_path.parent

    # Parent must exist and be a directory
    if not parent.exists() or not parent.is_dir():
        print("Error: invalid log file path", file=sys.stderr)
        sys.exit(1)

    # Parent must be writable
    if not os.access(parent, os.W_OK):
        print("Error: invalid log file path", file=sys.stderr)
        sys.exit(1)

    # The log file itself MUST ALREADY EXIST.
    if not log_path.exists():
        print("Error: invalid log file path", file=sys.stderr)
        sys.exit(1)

    # If it exists, verify it is a writable file
    if not log_path.is_file():
        print("Error: invalid log file path", file=sys.stderr)
        sys.exit(1)

    try:
        with open(log_path, "a"):
            pass
    except Exception:
        print("Error: invalid log file path", file=sys.stderr)
        sys.exit(1)

    token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        print("Error: invalid GitHub token", file=sys.stderr)
        sys.exit(1)

    try:
        r = requests.get(
            "https://api.github.com/user",
            headers={"Authorization": f"token {token}"},
            timeout=3,
        )
        if r.status_code != 200:
            print("Error: invalid GitHub token", file=sys.stderr)
            sys.exit(1)
    except Exception:
        print("Error: invalid GitHub token", file=sys.stderr)
        sys.exit(1)

    os.environ["LOG_LEVEL"] = str(args.log_level)
    setup_logging(level=args.log_level, json_lines=not args.log_text)

    run_id = set_run_id(args.run_id)
    log = get_logger("cli")

    import logging

    for h in logging.getLogger().handlers:
        if isinstance(h, logging.StreamHandler):
            h.stream = sys.stderr

    sys.stdout = _BOOT_STDOUT

    start_ns = time.perf_counter_ns()
    log.info("run started", extra={"phase": "run", "run_id": run_id})

    try:
        urls = read_urls(url_file_path)
    except Exception as e:
        print(f"Error reading URL file {e}", file=sys.stderr)
        log.exception("failed to read url file", extra={"phase": "run"})
        sys.exit(1)

    classifications = []
    for line in urls:
        line_classifications: dict[str, str] = {}

        for url in line:
            log.info("processing url", extra={"phase": "controller", "url": url})
            try:
                with redirect_stdout(io.StringIO()):
                    url_type = classify_url(url)
            except Exception:
                log.exception("classification failed", extra={"url": url})
                # Treat any classification failure as dataset
                url_type = "dataset"

            if url_type == "unknown":
                log.warning("unknown url type, treating as dataset")
                url_type = "dataset"

            line_classifications[url] = url_type

        if not line_classifications and line:
            # If everything failed, still score last URL as dataset.
            fallback_url = line[-1]
            line_classifications[fallback_url] = "dataset"

        classifications.append(line_classifications)

    # Now score each line
    for line in classifications:
        start_time = time.perf_counter_ns()
        try:
            name = "unknown-model"
            category = "model"

            net_score = 0.0
            ramp_up = 0.0
            bus_factor = 0.0
            performance_claims = 0.0
            license = 0.0
            dataset_and_code_score = 0.0
            dataset_quality = 0.0
            code_quality = 0.0

            net_score_latency = 0
            ramp_up_latency = 0
            bus_factor_latency = 0
            performance_claims_latency = 0
            license_latency = 0
            size_latency = 0
            dataset_and_code_score_latency = 0
            dataset_quality_latency = 0
            code_quality_latency = 0

            size_dict = {
                "raspberry_pi": 0.0,
                "jetson_nano": 0.0,
                "desktop_pc": 0.0,
                "aws_server": 0.0,
            }

            for url, url_type in line.items():
                try:
                    with redirect_stdout(io.StringIO()):
                        repo = get_repo_id(url, url_type) or ""
                except Exception:
                    log.exception("get_repo_id failed", extra={"url": url})
                    repo = ""

                parts = repo.split("/", 1)
                name = parts[1] if len(parts) == 2 else (parts[0] if parts else "")
                category = url_type.upper()

                if not name:
                    stripped = url.strip().strip("/")
                    if "/" in stripped:
                        candidate = stripped.rsplit("/", 1)[-1] or stripped
                    else:
                        candidate = stripped
                    name = candidate or "unknown-model"

                tasks = {}
                if url_type == "code":
                    tasks["code_quality"] = 0 #lambda: get_code_quality(url, url_type)
                elif url_type == "dataset":
                    tasks["dataset_quality"] = lambda: get_dataset_quality_score(
                        url, url_type
                    )
                    tasks["dataset_and_code_score"] = (
                        lambda: get_dataset_and_code_score(url, url_type)
                    )
                elif url_type == "model":
                    tasks["size"] = lambda: get_size_score(url, url_type)
                    tasks["license"] = lambda: get_license_score(url, url_type)
                    tasks["performance_claims"] = lambda: get_performance_claims(
                        url, url_type
                    )
                    tasks["bus_factor"] = lambda: get_bus_factor(url, url_type)
                    tasks["ramp_up"] = lambda: get_ramp_up(url, url_type)

                with redirect_stdout(io.StringIO()):
                    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
                        futures = {ex.submit(fn): mname for mname, fn in tasks.items()}
                        for fut in as_completed(futures):
                            metric_name = futures[fut]
                            try:
                                val, lat = fut.result()
                            except Exception:
                                log.exception(
                                    "metric failed",
                                    extra={"metric": metric_name, "url": url},
                                )
                                val, lat = (0.0, 0)

                            # Normalize latency in case a metric returns something odd.
                            try:
                                lat_int = int(lat)
                                if lat_int < 0:
                                    lat_int = 0
                            except Exception:
                                lat_int = 0

                            if metric_name == "code_quality":
                                code_quality = safe_metric_value(val)
                                code_quality_latency = lat_int
                            elif metric_name == "dataset_quality":
                                dataset_quality = safe_metric_value(val)
                                dataset_quality_latency = lat_int
                            elif metric_name == "dataset_and_code_score":
                                dataset_and_code_score = safe_metric_value(val)
                                dataset_and_code_score_latency = lat_int
                            elif metric_name == "size":
                                # Guard against size metric returning None / wrong type.
                                if isinstance(val, dict) and val is not None:
                                    size_dict = val
                                size_latency = lat_int
                            elif metric_name == "license":
                                license = safe_metric_value(val)
                                license_latency = lat_int
                            elif metric_name == "performance_claims":
                                performance_claims = safe_metric_value(val)
                                performance_claims_latency = lat_int
                            elif metric_name == "bus_factor":
                                bus_factor = safe_metric_value(val)
                                bus_factor_latency = lat_int
                            elif metric_name == "ramp_up":
                                ramp_up = safe_metric_value(val)
                                ramp_up_latency = lat_int

            if not line:
                output = {
                    "name": "unknown-model",
                    "category": "UNKNOWN",
                    "net_score": 0.0,
                    "net_score_latency": 0,
                    "ramp_up_time": 0.0,
                    "ramp_up_time_latency": 0,
                    "bus_factor": 0.0,
                    "bus_factor_latency": 0,
                    "performance_claims": 0.0,
                    "performance_claims_latency": 0,
                    "license": 0.0,
                    "license_latency": 0,
                    "size_score": {
                        "raspberry_pi": 0.0,
                        "jetson_nano": 0.0,
                        "desktop_pc": 0.0,
                        "aws_server": 0.0,
                    },
                    "size_score_latency": 0,
                    "dataset_and_code_score": 0.0,
                    "dataset_and_code_score_latency": 0,
                    "dataset_quality": 0.0,
                    "dataset_quality_latency": 0,
                    "code_quality": 0.0,
                    "code_quality_latency": 0,
                }
                print(json.dumps(output, separators=(",", ":")))
                sys.stdout.flush()
                continue

            # Compute net score using sanitized metric values.
            size_score = compute_size_score(size_dict)

            net_score = (
                SIZE_WEIGHT * size_score
                + LICENSE_WEIGHT * license
                + RAMP_UP_WEIGHT * ramp_up
                + BUS_FACTOR_WEIGHT * bus_factor
                + DATASET_QUALITY_WEIGHT * dataset_quality
                + CODE_QUALITY_WEIGHT * code_quality
                + PERFORMANCE_CLAIMS_WEIGHT * performance_claims
                + DATASET_AND_CODE_WEIGHT * dataset_and_code_score
            )

            net_score_latency = max(
                1, math.ceil((time.perf_counter_ns() - start_time) / 1_000_000)
            )

            output = {
                "name": name,
                "category": category,
                "net_score": safe_round(net_score),
                "net_score_latency": net_score_latency,
                "ramp_up_time": safe_round(ramp_up),
                "ramp_up_time_latency": ramp_up_latency,
                "bus_factor": safe_round(bus_factor),
                "bus_factor_latency": bus_factor_latency,
                "performance_claims": safe_round(performance_claims),
                "performance_claims_latency": performance_claims_latency,
                "license": safe_round(license),
                "license_latency": license_latency,
                "size_score": (
                    {k: safe_round(v) for k, v in size_dict.items()}
                    if isinstance(size_dict, dict) and size_dict
                    else {
                        "raspberry_pi": 0.0,
                        "jetson_nano": 0.0,
                        "desktop_pc": 0.0,
                        "aws_server": 0.0,
                    }
                ),
                "size_score_latency": size_latency,
                "dataset_and_code_score": safe_round(dataset_and_code_score),
                "dataset_and_code_score_latency": dataset_and_code_score_latency,
                "dataset_quality": safe_round(dataset_quality),
                "dataset_quality_latency": dataset_quality_latency,
                "code_quality": safe_round(code_quality),
                "code_quality_latency": code_quality_latency,
            }

            print(json.dumps(output, separators=(",", ":")))

        except Exception:
            log.exception("unexpected error while scoring line")
            print(
                json.dumps(
                    {
                        "name": "unknown-model",
                        "category": "model",
                        "net_score": 0.0,
                        "net_score_latency": 0,
                        "ramp_up_time": 0.0,
                        "ramp_up_time_latency": 0,
                        "bus_factor": 0.0,
                        "bus_factor_latency": 0,
                        "performance_claims": 0.0,
                        "performance_claims_latency": 0,
                        "license": 0.0,
                        "license_latency": 0,
                        "size_score": {
                            "raspberry_pi": 0.0,
                            "jetson_nano": 0.0,
                            "desktop_pc": 0.0,
                            "aws_server": 0.0,
                        },
                        "size_score_latency": 0,
                        "dataset_and_code_score": 0.0,
                        "dataset_and_code_score_latency": 0,
                        "dataset_quality": 0.0,
                        "dataset_quality_latency": 0,
                        "code_quality": 0.0,
                        "code_quality_latency": 0,
                    },
                    separators=(",", ":"),
                )
            )
            sys.stdout.flush()
            continue

    dur_ms = (time.perf_counter_ns() - start_ns) // 1_000_000
    log.info("run finished", extra={"phase": "run", "latency_ms": dur_ms})
    exit(0)


if __name__ == "__main__":
    main()
