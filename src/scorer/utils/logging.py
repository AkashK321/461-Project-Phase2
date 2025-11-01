# logging config
"""
Centralized logging for files with numeric verbosity levels:
0 = silent, 1 = info, 2 = debug
Respects:
  - $LOG_FILE  (path to log file; default: logs/scorer.log)
  - $LOG_LEVEL (0|1|2; default: 0)
"""

from __future__ import annotations
import contextvars
import json
import logging
import os
import time
import sys
import traceback
import uuid
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

_run_id_var = contextvars.ContextVar("run_id", default=None)
_url_var = contextvars.ContextVar("url", default=None)
_metric_var = contextvars.ContextVar("metric", default=None)


def set_run_id(run_id: Optional[str] = None) -> str:
    rid = run_id or str(uuid.uuid4())
    _run_id_var.set(rid)
    return rid


def set_url(url: Optional[str]) -> None:
    _url_var.set(url)


def set_metric(metric_name: Optional[str]) -> None:
    _metric_var.set(metric_name)


def _extra(**kw) -> Dict[str, Any]:
    ex = dict(run_id=_run_id_var.get(), url=_url_var.get(), metric=_metric_var.get())
    ex.update({k: v for k, v in kw.items() if v is not None})
    return ex


class JSONLineFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        ts = (
            datetime.fromtimestamp(record.created, tz=timezone.utc)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z")
        )
        payload = {
            "ts": ts,
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # pass-through extras if present
        for k in (
            "run_id",
            "url",
            "metric",
            "latency_ms",
            "function",
            "phase",
            "count",
            "file",
            "type",
        ):
            if hasattr(record, k):
                payload[k] = getattr(record, k)
            elif record.__dict__.get("extra", {}).get(k) is not None:
                payload[k] = record.__dict__["extra"][k]
        if record.exc_info:
            payload["exc"] = "".join(traceback.format_exception(*record.exc_info))[
                -4000:
            ]
        return json.dumps(payload, ensure_ascii=False)


class TextFormatter(logging.Formatter):
    def __init__(self):
        super().__init__(
            fmt="%(asctime)s | %(levelname)s | %(name)s | run=%(run_id)s "
            "url=%(url)s metric=%(metric)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )


def _fail_invalid_log_path(log_path: Path, err: BaseException) -> None:
    """
    Print a clear error to stderr and exit non-zero so the autograder
    can detect the invalid path case.
    """
    msg = f"Invalid log file path '{log_path}': {getattr(err, 'strerror', None) or err}"
    print(msg, file=sys.stderr)
    raise SystemExit(2)


def _is_running_in_aws_lambda() -> bool:
    """Check for an environment variable that is set by AWS Lambda."""
    return "AWS_LAMBDA_FUNCTION_NAME" in os.environ


def _setup_lambda_logging(level: int) -> None:
    """Configure logging to standard output for AWS Lambda."""
    logger = logging.getLogger("scorer")
    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONLineFormatter())
    logger.addHandler(handler)

def _validate_log_path(log_path: Path) -> None:
    """
    Ensure parent exists, the path is not a directory, and is writable.
    Touch the file (append mode) to verify.
    """
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if log_path.exists() and log_path.is_dir():
            raise IsADirectoryError(f"{log_path} is a directory")
        # Touch to verify writability
        with open(log_path, "a", encoding="utf-8"):
            pass
    except OSError as e:
        _fail_invalid_log_path(log_path, e)


def _normalize_verbosity(level: Optional[Union[int, str]]) -> int:
    """
    Return 0,1,2. Accepts ints, '0'/'1'/'2', or legacy words.
    """
    if level is None:
        s = os.environ.get("LOG_LEVEL", "0").strip()
    else:
        s = str(level).strip()

    # numeric path
    if s.isdigit():
        v = int(s)
        return 0 if v <= 0 else 2 if v >= 2 else 1

    # legacy word mapping (be forgiving)
    s_up = s.upper()
    if s_up == "DEBUG":
        return 2
    if s_up in {"INFO", "WARNING", "ERROR", "CRITICAL"}:
        return 1
    # default
    return 0


def _verbosity_to_logging_level(v: int) -> int:
    """
    Map 0/1/2 -> Python logging level.
      0 -> suppress everything (use a level above CRITICAL)
      1 -> INFO
      2 -> DEBUG
    """
    if v <= 0:
        return 100  # higher than CRITICAL; effectively silent but file still created
    return logging.INFO if v == 1 else logging.DEBUG


_INITIALIZED = False


def setup_logging(
    *, level: Optional[Union[int, str]] = None, json_lines: bool = True
) -> Path:
    """
    Initialize project logging to a rotating file.
    - File path from $LOG_FILE, else ./logs/scorer.log
    - Verbosity from $LOG_LEVEL (0|1|2); param 'level' overrides env
    - Rotates at 5 MB, keep 2 backups.
    - No console handler (stdout must remain clean for grader).
    """
    global _INITIALIZED
    
    # Normalize verbosity (0/1/2) and set env
    verbosity = _normalize_verbosity(level)
    os.environ["LOG_LEVEL"] = str(verbosity)
    py_level = _verbosity_to_logging_level(verbosity)

    if _is_running_in_aws_lambda():
        if not _INITIALIZED:
            _setup_lambda_logging(py_level)
            _INITIALIZED = True
        return Path("/tmp/lambda.log") # Return a dummy path

    # Ensure LOG_FILE is set or use default
    log_path = Path(os.environ.get("LOG_FILE", "logs/scorer.log"))
    os.environ.setdefault("LOG_FILE", str(log_path))

    if not log_path.exists() or log_path.is_dir():
        print(f"Error: log file {log_path} does not exist.", file=sys.stderr)
        sys.exit(1)

    if _INITIALIZED:
        # Even if already initialized, return the resolved path
        return Path(os.environ["LOG_FILE"]).resolve()

    _validate_log_path(log_path)

    logger = logging.getLogger("scorer")
    logger.setLevel(py_level)
    logger.propagate = False

    try:
        handler = RotatingFileHandler(
            log_path, maxBytes=5_000_000, backupCount=2, encoding="utf-8"
        )
    except OSError as e:
        print(f"Error: cannot open log file {log_path}: {e}", file=sys.stderr)
        sys.exit(1)

    handler.setLevel(py_level)
    handler.setFormatter(JSONLineFormatter() if json_lines else TextFormatter())
    logger.addHandler(handler)

    # Keep root logger quiet; we only use "scorer"
    logging.getLogger().handlers.clear()
    logging.getLogger().setLevel(100)

    _INITIALIZED = True
    return log_path.resolve()


def get_logger(name: Optional[str] = None) -> logging.LoggerAdapter:
    base = logging.getLogger("scorer" + ("" if not name else f".{name}"))
    return logging.LoggerAdapter(base, _extra())


def log_call(phase: str = "metric") -> Callable:
    def deco(fn: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            log = get_logger(fn.__module__)
            start = time.perf_counter_ns()
            log.info(
                f"start {fn.__name__}", extra=_extra(phase=phase, function=fn.__name__)
            )
            try:
                result = fn(*args, **kwargs)
                return result
            except Exception:
                dur_ms = (time.perf_counter_ns() - start) // 1_000_000
                log.exception(
                    f"error {fn.__name__}",
                    extra=_extra(phase=phase, function=fn.__name__, latency_ms=dur_ms),
                )
                raise
            finally:
                dur_ms = (time.perf_counter_ns() - start) // 1_000_000
                log.info(
                    f"end {fn.__name__}",
                    extra=_extra(phase=phase, function=fn.__name__, latency_ms=dur_ms),
                )

        return wrapper

    return deco
