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
from utils.logging import setup_logging, set_run_id, get_logger
from url_handler.base import classify_url
from urllib.parse import urlparse

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

    GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
    if not GITHUB_TOKEN:
        print(
            "Warning: GITHUB_TOKEN environment variable is not set or empty.",
            file=sys.stderr,
        )

    if args.log_file:
        os.environ["LOG_FILE"] = str(args.log_file)
    else:
        os.environ.setdefault("LOG_FILE", "logs/scorer.log")

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
        line_classifications = {}

        for url in line:
            log.info("processing url", extra={"phase": "controller", "url": url})
            try:
                with redirect_stdout(io.StringIO()):
                    url_type = classify_url(url)
            except Exception:
                log.exception("classification failed", extra={"url": url})

                # -------------------------------
                # ***** REQUIRED FIX HERE *****
                # On any failure, still treat as dataset
                url_type = "dataset"
                line_classifications[url] = url_type
                continue
                # -------------------------------

            if url_type == "unknown":
                log.warning("unknown url type, treating as dataset")
                url_type = "dataset"

            line_classifications[url] = url_type

        if not line_classifications and line:
            fallback_url = line[-1]
            line_classifications[fallback_url] = "dataset"

        classifications.append(line_classifications)

    # Now score each line
    for line in classifications:
        start_time = time.perf_counter_ns()
        try:
            name = "unknown-model"
            category = "MODEL"

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
                    tasks["code_quality"] = lambda: get_code_quality(url, url_type)
                elif url_type == "dataset":
                    tasks["dataset_quality"] = lambda: get_dataset_quality_score(
                        url, url_type
                    )
                    tasks["dataset_and_code_score"] = lambda: get_dataset_and_code_score(
                        url, url_type
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
                        futures = {ex.submit(fn): name for name, fn in tasks.items()}
                        for fut in as_completed(futures):
                            metric_name = futures[fut]
                            try:
                                val, lat = fut.result()
                            except Exception:
                                log.exception("metric failed", extra={"metric": metric_name})
                                val, lat = (0.0, 0)

                            if metric_name == "code_quality":
                                code_quality, code_quality_latency = val, lat
                            elif metric_name == "dataset_quality":
                                dataset_quality, dataset_quality_latency = val, lat
                            elif metric_name == "dataset_and_code_score":
                                dataset_and_code_score, dataset_and_code_score_latency = (
                                    val,
                                    lat,
                                )
                            elif metric_name == "size":
                                size_dict, size_latency = val, lat
                            elif metric_name == "license":
                                license, license_latency = val, lat
                            elif metric_name == "performance_claims":
                                performance_claims, performance_claims_latency = val, lat
                            elif metric_name == "bus_factor":
                                bus_factor, bus_factor_latency = val, lat
                            elif metric_name == "ramp_up":
                                ramp_up, ramp_up_latency = val, lat

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

            size_score = sum(size_dict.values()) / len(size_dict)

            net_score = (
                0.15 * size_score
                + 0.15 * license
                + 0.10 * ramp_up
                + 0.10 * bus_factor
                + 0.15 * dataset_quality
                + 0.10 * code_quality
                + 0.15 * performance_claims
                + 0.10 * dataset_and_code_score
            )

            net_score_latency = max(
                1, math.ceil((time.perf_counter_ns() - start_time) / 1_000_000)
            )

            output = {
                "name": name,
                "category": category,
                "net_score": round(net_score, 2),
                "net_score_latency": net_score_latency,
                "ramp_up_time": round(ramp_up, 2),
                "ramp_up_time_latency": ramp_up_latency,
                "bus_factor": round(bus_factor, 2),
                "bus_factor_latency": bus_factor_latency,
                "performance_claims": round(performance_claims, 2),
                "performance_claims_latency": performance_claims_latency,
                "license": round(license, 2),
                "license_latency": license_latency,
                "size_score": {k: round(v, 2) for k, v in size_dict.items()},
                "size_score_latency": size_latency,
                "dataset_and_code_score": round(dataset_and_code_score, 2),
                "dataset_and_code_score_latency": dataset_and_code_score_latency,
                "dataset_quality": round(dataset_quality, 2),
                "dataset_quality_latency": dataset_quality_latency,
                "code_quality": round(code_quality, 2),
                "code_quality_latency": code_quality_latency,
            }

            print(json.dumps(output, separators=(",", ":")))

        except Exception:
            log.exception("unexpected error while scoring line")
            print(
                json.dumps(
                    {
                        "name": "unknown-model",
                        "category": "MODEL",
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
