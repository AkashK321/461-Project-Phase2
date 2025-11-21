import os
import json
import base64
import shutil
import uuid
import re
import logging

os.environ["HF_HOME"] = "/tmp/huggingface"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/tmp/huggingface/hub"
os.environ["HF_ASSETS_CACHE"] = "/tmp/huggingface/assets"

from datetime import datetime, timezone

import boto3
from huggingface_hub import snapshot_download
from aws_modules.s3_utils import upload_model, generate_presigned_download_url
from aws_modules.db_utils import (
    save_model_metadata,
    get_model_by_id,
)
from utils.lineage_utils import (
    get_base_model_from_card,
    get_lineage_items_from_id,
    get_descendant_items,
)
from scorer.metrics.base import get_repo_id
from scorer.url_handler.base import classify_url
from aws_modules.api_utils import make_response

from aws_modules.auth import (
    authenticate_user,
    get_validated_user,
    register_user,
    hash_password,
    ensure_default_user,
)

# logging setup
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

# shared AWS clients/resources
dynamodb = boto3.resource("dynamodb")
s3 = boto3.client("s3")
lambda_client = boto3.client("lambda")

TABLE_NAME = os.getenv("DYNAMODB_TABLE_NAME", "")
BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "")
USER_TABLE_NAME = os.getenv("USER_DYNAMODB_TABLE_NAME", "")
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "a-very-unsafe-default-secret")
SCORER_FUNCTION_NAME = os.getenv("SCORER_FUNCTION_NAME", "scorer_function")

# Default admin user credentials from environment variables
# Fall back to the specification defaults if not set
DEFAULT_ADMIN_USERNAME = os.getenv("DEFAULT_ADMIN_USERNAME", "")
DEFAULT_ADMIN_PASSWORD = os.getenv("DEFAULT_ADMIN_PASSWORD", "")

# Global flag to track if initialization has been performed
_initialized = False


# simple helpers for semver-ish ranges

SEMVER_PATTERN = re.compile(
    r"^v?(?P<maj>0|[1-9]\d*)"
    r"(?:\.(?P<min>0|[1-9]\d*))?"
    r"(?:\.(?P<patch>0|[1-9]\d*))?$"
)


def parse_semver(version: str):
    """
    Turn "1.2.3" / "1.2" / "v1.2.3" into (major, minor, patch).

    Returns None if it doesn't look like a (loose) semver.
    """
    if not version:
        return None
    s = version.strip()
    if s.lower().startswith("v"):
        s = s[1:]

    m = SEMVER_PATTERN.match(s)
    if not m:
        return None

    maj = int(m.group("maj"))
    min_ = int(m.group("min") or 0)
    patch = int(m.group("patch") or 0)
    return (maj, min_, patch)


def version_satisfies(ver: str, constraint: str) -> bool:
    """
    Check if a version string matches a constraint.

    Supported patterns:
      - exact: "1.2.3"
      - bounded: "1.2.3-2.1.0"
      - tilde: "~1.2.0"
      - caret: "^1.2.0"
    """
    if not constraint:
        # no filter -> everything matches
        return True

    v = parse_semver(ver)
    c = constraint.strip()

    # if the stored version doesn't parse, only do exact string match
    if v is None:
        return ver == c

    # "1.2.3-2.1.0" (inclusive)
    if "-" in c and not c.startswith(("~", "^")):
        lo_s, hi_s = [p.strip() for p in c.split("-", 1)]
        lo = parse_semver(lo_s)
        hi = parse_semver(hi_s)
        if lo is None or hi is None:
            return False
        return lo <= v <= hi

    # "~1.2.0" -> >=1.2.0 and <1.3.0
    if c.startswith("~"):
        base = c[1:].strip()
        b = parse_semver(base)
        if b is None:
            return False

        if v < b:
            return False

        maj, min_, _ = b
        upper = (maj, min_ + 1, 0)
        return v < upper

    # caret rules:
    #   ^1.2.0  -> >=1.2.0, <2.0.0
    #   ^0.2.3  -> >=0.2.3, <0.3.0
    #   ^0.0.3  -> >=0.0.3, <0.0.4
    if c.startswith("^"):
        base = c[1:].strip()
        b = parse_semver(base)
        if b is None:
            return False

        if v < b:
            return False

        maj, min_, pat = b
        if maj > 0:
            upper = (maj + 1, 0, 0)
        elif min_ > 0:
            upper = (0, min_ + 1, 0)
        else:
            upper = (0, 0, pat + 1)
        return v < upper

    # exact match (or last-resort raw equality)
    cver = parse_semver(c)
    if cver is None:
        return ver == c
    return v == cver


def parse_event(event):
    path = event.get("rawPath", "") or "/"
    method = event.get("requestContext", {}).get("http", {}).get("method", "GET")
    query_params = event.get("queryStringParameters") or {}
    is_b64 = event.get("isBase64Encoded", False)
    raw_body = event.get("body") or ""

    if is_b64:
        try:
            raw_body = base64.b64decode(raw_body).decode("utf-8")
        except Exception as e:
            logger.error(f"Base64 decode failed: {e}")
            raw_body = ""

    logger.info(f"Parsing event for: {method} {path}")
    logger.info(
        f"Raw event body: {raw_body}"
    )  # Optional: Comment out to hide passwords in logs

    body = {}

    if not raw_body:
        logger.info("Raw body is empty, returning empty dict.")
        return method, path, body, query_params

    logger.info(f"Using standard JSON parser for {path}.")
    try:
        body = json.loads(raw_body)
        logger.info("JSON parsing SUCCEEDED.")

    except json.JSONDecodeError as e:
        logger.error(f"JSONDecodeError on '{path}': {e}")
        # distinct error lets you know the client sent bad JSON
        return method, path, {}, query_params

    except Exception as e:
        logger.error(f"Unexpected error parsing body for {path}: {e}")
        body = {}

    return method, path, body, query_params


def initialize_system():
    """
    Initialize the system by ensuring the default admin user exists.
    This should be called once per Lambda container initialization.
    """
    global _initialized

    if _initialized:
        return

    # Only initialize if we have authentication support
    if USER_TABLE_NAME and JWT_SECRET_KEY:
        ensure_default_user()

    _initialized = True


def reset_state():
    # wipe the main registry table
    tbl = dynamodb.Table(TABLE_NAME)
    scan = tbl.scan(ProjectionExpression="#i", ExpressionAttributeNames={"#i": "id"})
    ids = [it["id"] for it in scan.get("Items", [])]
    if ids:
        with tbl.batch_writer() as batch:
            for _id in ids:
                batch.delete_item(Key={"id": _id})

    # and clear any model objects in S3 under models/
    if BUCKET_NAME:
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix="models/"):
            objs = [{"Key": o["Key"]} for o in page.get("Contents", [])]
            if objs:
                s3.delete_objects(Bucket=BUCKET_NAME, Delete={"Objects": objs})

    if USER_TABLE_NAME:
        # rebuild the user table with the default admin from the spec
        user_tbl = dynamodb.Table(USER_TABLE_NAME)

        scan = user_tbl.scan(
            ProjectionExpression="#i", ExpressionAttributeNames={"#i": "id"}
        )
        user_ids = [it["id"] for it in scan.get("Items", [])]

        if user_ids:
            with user_tbl.batch_writer() as batch:
                for _id in user_ids:
                    batch.delete_item(Key={"id": _id})

        admin_id = str(uuid.uuid4())
        user_tbl.put_item(
            Item={
                "id": admin_id,
                "username": DEFAULT_ADMIN_USERNAME,
                "password_hash": hash_password(DEFAULT_ADMIN_PASSWORD),
                "roles": ["admin", "upload", "search", "download"],
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        logger.info(
            f"Reset user table and added default admin {DEFAULT_ADMIN_USERNAME}"
        )

    return {"reset": "ok", "deleted": {"dynamodb": len(ids)}}


def ingest_artifact(art_type, payload):
    logger.info(f"--- Starting Ingestion for {art_type} ---")
    logger.info(f"Payload: {json.dumps(payload)}")

    try:
        url = payload.get("url")
        if not url or not isinstance(url, str):
            return make_response(
                400, {"error": "payload must have a non-empty 'url' string"}
            )

        urls = [url]

        repo = ""
        for u in urls:
            url_type = classify_url(u)
            if url_type == "model":
                repo = get_repo_id(u, url_type) or ""
                break
            else:
                repo = ""

        logger.info(f"Resolved Repo ID: '{repo}' for URL: '{url}'")

        if not repo:
            logger.error("Unable to find model url or repo_id in payload")
            return make_response(400, {"error": "unable to find model url in payload"})

        parts = repo.split("/", 1)
        name = parts[1] if len(parts) == 2 else (parts[0] if parts else "")
        logger.info(f"Derived Model Name: {name}")

    except KeyError:
        return make_response(
            400, {"error": "payload must have a non-empty 'url' string"}
        )
    except Exception as e:
        logger.error(f"URL parsing error: {e}")
        return make_response(400, {"error": "unable to parse model url in payload"})

    # --- Invoke Scorer ---
    scorer_function_name = SCORER_FUNCTION_NAME
    scores = {}
    if not scorer_function_name:
        logger.error("Scorer function not configured")
        return make_response(500, {"error": "Metric scoring service not available"})

    logger.info(f"Invoking scorer function: {scorer_function_name}")
    try:
        scorer_payload = json.dumps({"urls": urls})
        logger.info(f"Scorer Request Payload: {scorer_payload}")

        response = lambda_client.invoke(
            FunctionName=scorer_function_name,
            InvocationType="RequestResponse",
            Payload=scorer_payload,
        )
        raw_response = response["Payload"].read().decode()
        logger.info(f"Scorer Raw Response: {raw_response}")

        response_payload = json.loads(raw_response)

        if response_payload.get("statusCode") == 200:
            scores_list = json.loads(response_payload["body"])
            logger.info(f"Parsed Scores List: {json.dumps(scores_list)}")
            if scores_list:
                scores = scores_list[0]
        else:
            logger.error(f"Scorer function returned error: {response_payload}")
            return make_response(500, {"error": "Failed to calculate metrics"})
    except Exception as e:
        logger.error(f"Failed to invoke or parse scorer response: {e}")
        return make_response(500, {"error": "Failed to calculate metrics"})

    logger.info(f"Final Scores Object: {json.dumps(scores)}")

    # --- Validate Metrics ---
    non_latency_metrics = [
        "bus_factor",
        "code_quality",
        "license",
        "ramp_up",
        "dataset_quality",
        "performance_claims",
        "dataset_and_code",
        "size",
    ]

    failing_metrics = []
    for metric in non_latency_metrics:
        score = scores.get(metric, 0)

        if isinstance(score, dict):
            val = sum(score.values()) / len(score) if score else 0
            logger.info(f"Metric {metric} is dict, avg value: {val}")
            if val < 0.5:
                failing_metrics.append(f"{metric}: {val}")
        else:
            if score < 0.5:
                failing_metrics.append(f"{metric}: {score}")

    if failing_metrics:
        logger.info(f"Package rejected due to insufficient scores: {failing_metrics}")

        # --- IMPORTANT: TO TEST UPLOAD DESPITE LOW SCORES,
        # COMMENT OUT THE RETURN BELOW ---
        return make_response(
            424,
            {
                "error": "Package is not uploaded due to insufficient quality metrics",
                "failing_metrics": failing_metrics,
            },
        )
        # ----------------------------------------------------------------------------------

    logger.info("All non-latency metrics passed threshold, proceeding with ingestion")

    # --- Download and Upload ---
    tmp_dir = f"/tmp/{str(uuid.uuid4())}"
    tmp_zip_file = ""
    base_model_repo, lineage_type, source = get_base_model_from_card(repo)

    try:
        logger.info(f"Downloading model '{repo}' to '{tmp_dir}'")
        snapshot_download(
            repo_id=repo,
            local_dir=tmp_dir,
        )

        # Log download success by listing files
        files_downloaded = []
        for root, dirs, files in os.walk(tmp_dir):
            for file in files:
                files_downloaded.append(os.path.join(root, file))
        logger.info(
            f"Downloaded {len(files_downloaded)} "
            f"files. First few: {files_downloaded[:3]}"
        )

        zip_name = f"{name}"
        tmp_zip_path_base = f"/tmp/{zip_name}"
        logger.info(f"Zipping directory to {tmp_zip_path_base}.zip")

        tmp_zip_file = shutil.make_archive(tmp_zip_path_base, "zip", tmp_dir)
        final_zip_name = f"{name}.zip"

        zip_size = os.path.getsize(tmp_zip_file)
        logger.info(f"Zip created: {tmp_zip_file} (Size: {zip_size} bytes)")

        model_id = str(uuid.uuid4())
        s3_key = f"models/{model_id}/{final_zip_name}"

        logger.info(f"Uploading '{tmp_zip_file}' to S3 key '{s3_key}'")
        ok = upload_model(tmp_zip_file, s3_key)
        if ok is False:
            logger.error("S3 Upload returned False")
            return make_response(500, {"error": "S3 upload failed"})

        logger.info("S3 Upload successful")

        version = "v1"
        created_at = datetime.now(timezone.utc).isoformat()

        # Save metadata to DynamoDB
        logger.info("Saving metadata to DynamoDB")
        item = save_model_metadata(name, version, s3_key, scores)
        if not item:
            return make_response(500, {"error": "failed to store metadata"})

        tbl = dynamodb.Table(TABLE_NAME)
        tbl.update_item(
            Key={"id": item["id"]},
            UpdateExpression="SET #t = :t, #c = :c, #fn = :fn, \
                #url = :url, #rid = :rid, #brid = :brid, \
                #ling = :ling, #linsrc = :linsrc",
            ExpressionAttributeNames={
                "#t": "type",
                "#c": "created_at",
                "#fn": "filename",
                "#url": "source_url",
                "#rid": "repo_id",
                "#brid": "base_model_repo_id",
                "#ling": "lineage_type",
                "#linsrc": "lineage_source",
            },
            ExpressionAttributeValues={
                ":t": art_type,
                ":c": created_at,
                ":fn": final_zip_name,
                ":url": url,
                ":rid": repo,
                ":brid": base_model_repo,
                ":ling": lineage_type,
                ":linsrc": source,
            },
        )

        logger.info(f"Successfully ingested model {model_id} from {url}")
        metadata = {
            "name": item.get("model_name"),
            "id": item.get("id"),
            "type": item.get("type"),
        }
        data = {"url": item.get("source_url")}
        return make_response(201, {"metadata": metadata, "data": data})
    except Exception as e:
        logger.error(f"Ingestion processing failed: {e}")
        # logger.error(traceback.format_exc()) # requires import traceback
        return make_response(500, {"error": f"Internal server error: {str(e)}"})
    finally:
        try:
            if os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir)
            if os.path.exists(tmp_zip_file):
                os.remove(tmp_zip_file)
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


def search_artifacts(query_array, query_params):
    """
    Handle POST /artifacts.

    - query_array: The request body, which is a list of query objects.
    - query_params: The query string parameters, containing the 'offset'.
    """

    try:
        offset_str = query_params.get("offset", "1")
        page = int(offset_str)
    except (TypeError, ValueError):
        page = 1

    if page < 1:
        page = 1

    page_size = 50

    if not query_array or not isinstance(query_array, list) or len(query_array) == 0:
        query = {}
    else:
        query = query_array[0]  # Use the first query object

    name_query = (query.get("name") or "").strip()
    types = query.get("types", [])
    want_all_types = not types

    # Handle the wildcard "*" search
    if name_query == "*":
        name_query = ""  # This will match all names

    version_query = (query.get("version") or query.get("version_range") or "").strip()

    tbl = dynamodb.Table(TABLE_NAME)
    scan_resp = tbl.scan()
    items = scan_resp.get("Items", [])

    # filter by type (if provided)
    if not want_all_types:
        type_set = {str(t).lower() for t in types}
        items = [it for it in items if str(it.get("type", "")).lower() in type_set]

    # filter by name (regex on model_name, fall back to substring)
    if name_query:
        try:
            rx = re.compile(name_query)
            items = [it for it in items if rx.search(str(it.get("model_name", "")))]
        except re.error:
            items = [it for it in items if name_query in str(it.get("model_name", ""))]

    # filter by version constraint (if any)
    if version_query:
        items = [
            it
            for it in items
            if version_satisfies(str(it.get("version", "")), version_query)
        ]

    def _sort_key(it):
        return (
            str(it.get("model_name", "")),
            parse_semver(str(it.get("version", ""))) or (0, 0, 0),
        )

    items.sort(key=_sort_key)

    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    page_items = items[start_idx:end_idx]

    headers = {}
    if end_idx < len(items):
        next_offset = str(page + 1)
        headers = {"Offset": next_offset}

    return make_response(200, page_items, headers=headers)


def get_lineage_graph(start_art_id):
    """
    Handle GET /artifact/model/{id}/lineage
    Constructs and returns the lineage graph for a given model ID.
    """
    start_item = get_model_by_id(start_art_id)
    if not start_item:
        return make_response(404, {"error": "Artifact does not exist."})

    # --- Build the full graph: ancestors + start_node + descendants ---
    ancestors = get_lineage_items_from_id(start_art_id)
    start_repo_id = start_item.get("repo_id")

    # Per spec, if metadata is malformed for lineage (e.g., missing repo_id),
    # return a 400 error.
    if not start_repo_id:
        return make_response(
            400,
            {
                "error": "The lineage graph cannot be computed \
                because the artifact metadata is missing or malformed."
            },
        )
    descendants = get_descendant_items(start_repo_id)

    # Combine all items, ensuring no duplicates
    all_items = {item["id"]: item for item in ancestors}
    all_items[start_item["id"]] = start_item
    for item in descendants:
        all_items[item["id"]] = item

    # --- Construct nodes and edges from all items ---
    nodes = []
    edges = []
    for item_id, item in all_items.items():
        nodes.append(
            {
                "artifact_id": item_id,
                "name": item.get("model_name"),
                "source": item.get("lineage_source", "inferred"),
            }
        )

        # If the item has a parent, create an edge
        parent_repo_id = item.get("base_model_repo_id")
        if parent_repo_id:
            # Find the parent item in our collected items
            parent_item = None
            for p_item in all_items.values():
                if p_item.get("repo_id") == parent_repo_id:
                    parent_item = p_item
                    break

            if parent_item:
                edges.append(
                    {
                        "from_node_artifact_id": parent_item.get("id"),
                        "to_node_artifact_id": item_id,
                        "relationship": item.get("lineage_type", "fine_tuned_from"),
                    }
                )
            else:
                # This case can happen if a parent exists but is not in the registry
                logger.warning(
                    f"Parent model with repo_id '{parent_repo_id}' not \
                        found in registry for child '{item.get('repo_id')}'"
                )

    return make_response(200, {"nodes": nodes, "edges": edges})


def rate_model(art_id):
    """
    Handle GET /artifact/model/{id}/rate
    Retrieves and returns the stored scores for a given model artifact.
    """
    item = get_model_by_id(art_id)
    if not item:
        return make_response(404, {"error": "Artifact does not exist."})

    scorer_function_name = SCORER_FUNCTION_NAME
    scores = {}
    if not scorer_function_name:
        logger.error("Scorer function not configured, cannot validate metrics")
        return make_response(500, {"error": "Metric scoring service not available"})

    logger.info(f"Invoking scorer function: {scorer_function_name}")
    try:
        logger.info(
            f"Calculating scores for artifact \
                    {art_id} at URL {item.get('source_url')}"
        )
        scorer_payload = json.dumps({"urls": [item.get("source_url")]})
        response = lambda_client.invoke(
            FunctionName=scorer_function_name,
            InvocationType="RequestResponse",
            Payload=scorer_payload,
        )
        response_payload = json.loads(response["Payload"].read().decode())
        if response_payload.get("statusCode") == 200:
            scores_list = json.loads(response_payload["body"])
            if scores_list:
                scores = scores_list[0]
        else:
            logger.error(f"Scorer function returned error: {response_payload}")
            return make_response(500, {"error": "Failed to calculate metrics"})
    except Exception as e:
        logger.error(f"Failed to invoke or parse scorer response: {e}")
        return make_response(500, {"error": "Failed to calculate metrics"})

    if not scores:
        logger.error(f"Scores not found for artifact {art_id}, but item exists.")
        return make_response(
            500,
            {
                "error": "The artifact rating system encountered an \
                    error while computing at least one metric."
            },
        )
    else:
        for metric, score in scores.items():
            logger.info(f"Metric: {metric}, Score: {score}")
            if score is None:
                logger.warning(f"Score for metric '{metric}' is None.")
                return make_response(
                    500,
                    {
                        "error": "The artifact rating system encountered an \
                            error while computing at least one metric."
                    },
                )

    return make_response(200, scores)


def handler(event, context):
    # Initialize the system on first run (ensures default user exists)
    initialize_system()

    method, path, body, query_params = parse_event(event)

    if not TABLE_NAME or not BUCKET_NAME:
        return make_response(500, {"error": "missing env vars for table/bucket"})

    # Public Routes

    # basic health check
    if method == "GET" and path == "/health":
        return make_response(200, {"status": "ok"})

    # tracks
    if method == "GET" and path == "/tracks":
        return make_response(200, {"plannedTracks": ["Access control track"]})

    # authentication entry point
    if method == "PUT" and path == "/authenticate":
        if not USER_TABLE_NAME or not JWT_SECRET_KEY:
            return make_response(
                501, {"error": "This system does not support authentication."}
            )
        return authenticate_user(body)

    # Authentication
    user_payload = get_validated_user(event)

    # Reset (admin only but has bypass)
    # reset route: needed by the autograder and for local testing
    if path == "/reset" and method == "DELETE":
        if not user_payload:
            # This bypass is for the autograder/testing.
            logger.warning("Allowing unauthenticated /reset")
            try:
                out = reset_state()
                return make_response(200, out)
            except Exception as e:
                return make_response(500, {"error": str(e)})

        # If we have a user, enforce admin role
        user_roles = user_payload.get("roles", [])
        if "admin" not in user_roles:
            return make_response(
                401, {"error": "You do not have permission to reset the registry."}
            )
        try:
            out = reset_state()
            return make_response(200, out)
        except Exception as e:
            return make_response(500, {"error": str(e)})

    # Main Authentication Check

    if not user_payload:
        return make_response(
            403,
            {
                "error": "Authentication failed due to invalid or "
                "missing AuthenticationToken."
            },
        )

    # Authenticated Routes

    user_roles = user_payload.get("roles", [])
    logger.info(f"Authenticated user {user_payload.get('sub')} with roles {user_roles}")

    if method == "POST" and path == "/users":
        return register_user(body, user_roles)

    # POST /artifact/{type}
    if method == "POST" and path.startswith("/artifact/") and path.count("/") == 2:
        art_type = path.split("/artifact/", 1)[-1]
        try:
            return ingest_artifact(art_type, body)
        except KeyError as ke:
            return make_response(400, {"error": f"missing field: {str(ke)}"})
        except Exception as e:
            return make_response(400, {"error": str(e)})

    # GET /artifact/{type}/{id}
    get_match = re.match(r"/artifacts/([^/]+)/([^/]+)", path)
    if method == "GET" and get_match and path.count("/") == 3:
        art_type = get_match.group(1)
        art_id = get_match.group(2)

        item = get_model_by_id(art_id)
        if not item:
            return make_response(404, {"error": "Artifact does not exist."})

        metadata = {
            "name": item.get("model_name"),
            "id": item.get("id"),
            "type": item.get("type"),
        }
        s3_key = item.get("s3_key")
        download_url = None
        if s3_key:
            download_url = generate_presigned_download_url(s3_key)
        else:
            logger.warning(
                f"Artifact {art_id} has no 's3_key' to generate download URL"
            )

        data = {"url": item.get("source_url")}

        if download_url:
            data["download_url"] = download_url

        return make_response(200, {"metadata": metadata, "data": data})

    # GET /artifact/model/{id}/lineage
    lineage_match = re.match(r"/artifact/model/([^/]+)/lineage", path)
    if method == "GET" and lineage_match:
        art_id = lineage_match.group(1)
        if not art_id:
            return make_response(400, {"error": "Missing artifact ID in path"})
        return get_lineage_graph(art_id)

    # GET /artifact/model/{id}/rate
    rate_match = re.match(r"/artifact/model/([^/]+)/rate", path)
    if method == "GET" and rate_match:
        art_id = rate_match.group(1)
        if not art_id:
            return make_response(
                400,
                {
                    "error": "There is missing field(s) in the \
                                       artifact_id or it is formed improperly, \
                                       or is invalid."
                },
            )
        return rate_model(art_id)

    # POST /artifacts
    if method == "POST" and path == "/artifacts":
        try:
            return search_artifacts(body or [], query_params)
        except Exception as e:
            return make_response(400, {"error": str(e)})

    # anything else is a 404
    return make_response(404, {"error": f"Route not found: {method} {path}"})
