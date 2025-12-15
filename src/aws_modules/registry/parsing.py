"""
Parsing utilities for semantic versioning and API events.
"""

import json
import base64
import re


SEMVER_PATTERN = re.compile(
    r"^v?(?P<maj>0|[1-9]\d*)"
    r"(?:\.(?P<min>0|[1-9]\d*))?"
    r"(?:\.(?P<patch>0|[1-9]\d*))?$"
)


def _timeout_handler(signum, frame):
    raise TimeoutError("Regex execution timed out")


def parse_semver(version: str):
    """Parse a semantic version string into a tuple of integers.

    :param version: The version string to parse.
    :return: A tuple (major, minor, patch) or None if invalid.
    """
    if not version:
        return None
    s = version.strip()
    if s.lower().startswith("v"):
        s = s[1:]
    m = SEMVER_PATTERN.match(s)
    if not m:
        return None
    return (int(m.group("maj")), int(m.group("min") or 0), int(m.group("patch") or 0))


def version_satisfies(ver: str, constraint: str) -> bool:
    """Check if a version satisfies a semantic version constraint.

    :param ver: The version string to check.
    :param constraint: The constraint string (e.g., "^1.0.0", "~1.0").
    :return: True if the version satisfies the constraint, False otherwise.
    """
    if not constraint:
        return True
    v = parse_semver(ver)
    c = constraint.strip()
    if v is None:
        return ver == c
    if "-" in c and not c.startswith(("~", "^")):
        lo_s, hi_s = [p.strip() for p in c.split("-", 1)]
        lo, hi = parse_semver(lo_s), parse_semver(hi_s)
        return False if (lo is None or hi is None) else (lo <= v <= hi)
    if c.startswith("~"):
        b = parse_semver(c[1:].strip())
        if b is None:
            return False
        maj, min_, _ = b
        return v >= b and v < (maj, min_ + 1, 0)
    if c.startswith("^"):
        b = parse_semver(c[1:].strip())
        if b is None:
            return False
        maj, min_, pat = b
        upper = (
            (maj + 1, 0, 0)
            if maj > 0
            else ((0, min_ + 1, 0) if min_ > 0 else (0, 0, pat + 1))
        )
        return v >= b and v < upper
    cver = parse_semver(c)
    return v == cver if cver else ver == c


def parse_event(event):
    """Parse an API Gateway event into method, path, body, and query params.

    :param event: The event dictionary from API Gateway.
    :return: A tuple (method, path, body, query_params).
    """
    path = event.get("rawPath", "") or event.get("path") or "/"
    method = event.get("requestContext", {}).get("http", {}).get(
        "method", "GET"
    ) or event.get("httpMethod", "GET")
    query_params = event.get("queryStringParameters") or {}
    is_b64 = event.get("isBase64Encoded", False)
    raw_body = event.get("body") or ""
    if is_b64:
        try:
            raw_body = base64.b64decode(raw_body).decode("utf-8")
        except Exception:
            raw_body = ""
    body = {}
    if raw_body:
        try:
            body = json.loads(raw_body)
        except Exception:
            pass
    return method, path, body, query_params
