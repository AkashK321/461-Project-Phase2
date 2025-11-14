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
from aws_modules.s3_utils import upload_model
from aws_modules.db_utils import save_model_metadata, get_model_by_id, get_model_by_repo_id
from utils.lineage_utils import get_base_model_from_card, get_lineage_items_from_id, get_descendant_items
from scorer.metrics.base import get_repo_id
from scorer.url_handler.base import classify_url
from aws_modules.api_utils import make_response

from aws_modules.auth import (
    authenticate_user,
    get_validated_user,
    register_user,
    hash_password,
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
    # pull out method, path, and JSON body from the API Gateway event
    path = event.get("rawPath", "") or "/"
    method = event.get("requestContext", {}).get("http", {}).get("method", "GET")
    is_b64 = event.get("isBase64Encoded", False)
    raw_body = event.get("body") or "{}"
    if is_b64:
        raw_body = base64.b64decode(raw_body).decode("utf-8")
    try:
        body = json.loads(raw_body) if raw_body else {}
    except Exception:
        body = {}
    return method, path, body


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
        admin_user = "ece30861defaultadminuser"
        admin_pass = "correcthorsebatterystaple123(!__+@**(A;DROP TABLE packages"

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
                "username": admin_user,
                "password_hash": hash_password(admin_pass),
                "roles": ["admin", "upload", "search", "download"],
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        logger.info(f"Reset user table and added default admin {admin_user}")

    return {"reset": "ok", "deleted": {"dynamodb": len(ids)}}


def ingest_artifact(art_type, payload):
    """
    Handle POST /artifact/{type}.

    Payload should have:
      - urls: non-empty list of URLs pointing at the model.
    """
    try:
        if type(payload["urls"]) is not list or len(payload["urls"]) == 0:
            return make_response(
                400, {"error": "payload must have non-empty 'urls' list"}
            )
        for url in payload["urls"]:
            url_type = classify_url(url)
            if url_type == "model":
                repo = get_repo_id(url, url_type) or ""
                break
            else:
                repo = ""
        parts = repo.split("/", 1)
        name = parts[1] if len(parts) == 2 else (parts[0] if parts else "")
    except Exception:
        return make_response(400, {"error": "unable to find model url in payload"})

    # per the spec we eventually need a pre-metric gate here
    # (non-latency metrics >= 0.5 before we allow ingestion).
    logger.warning("Metric pre-check not implemented. Proceeding directly to download.")

    tmp_dir = f"/tmp/{str(uuid.uuid4())}"
    tmp_zip_file = ""
    base_model_repo, lineage_type, source = get_base_model_from_card(repo)
    logger.info(f"Base model repo from card: {base_model_repo}-{lineage_type}")

    try:
        # pull the full repo contents down locally
        logger.info(f"Downloading model '{repo}' to '{tmp_dir}'")
        snapshot_download(
            repo_id=repo,
            local_dir=tmp_dir,
        )

        # zip up what we just downloaded
        zip_name = f"{name}"
        tmp_zip_path_base = f"/tmp/{zip_name}"
        tmp_zip_file = shutil.make_archive(tmp_zip_path_base, "zip", tmp_dir)
        final_zip_name = f"{name}.zip"

        # push to S3 and record metadata
        model_id = str(uuid.uuid4())
        s3_key = f"models/{model_id}/{final_zip_name}"

        logger.info(f"Uploading '{tmp_zip_file}' to S3 key '{s3_key}'")
        ok = upload_model(tmp_zip_file, s3_key)
        if ok is False:
            return make_response(500, {"error": "S3 upload failed"})

        version = "v1"  # could be pulled from HF metadata later

        # --- Invoke scorer_function to get scores ---
        scorer_function_name = SCORER_FUNCTION_NAME
        scores = {}
        if scorer_function_name:
            logger.info(f"Invoking scorer function: {scorer_function_name}")
            try:
                scorer_payload = json.dumps({"urls": [url]})
                response = lambda_client.invoke(
                    FunctionName=scorer_function_name,
                    InvocationType="RequestResponse",
                    Payload=scorer_payload,
                )
                response_payload = json.loads(response["Payload"].read().decode())
                if response_payload.get("statusCode") == 200:
                    scores_list = json.loads(response_payload["body"])
                    if scores_list:
                        scores = scores_list[0]  # We only sent one URL
            except Exception as e:
                logger.error(f"Failed to invoke or parse scorer response: {e}")

        created_at = datetime.now(timezone.utc).isoformat()

        item = save_model_metadata(name, version, s3_key, scores)
        if not item:
            return make_response(500, {"error": "failed to store metadata"})

        # attach extra fields the base helper doesn't know about
        tbl = dynamodb.Table(TABLE_NAME)
        tbl.update_item(
            Key={"id": item["id"]},
            UpdateExpression="SET #t = :t, #c = :c, #fn = :fn, \
                #url = :url, #rid = :rid, #brid = :brid, #ling = :ling, #linsrc = :linsrc",
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
        return make_response(201, {"id": item["id"], "s3_key": s3_key, "model": name})
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        return make_response(500, {"error": f"Internal server error: {str(e)}"})
    finally:
        # keep /tmp from filling up between invocations
        try:
            if os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir)
                logger.info(f"Cleaned up temp dir: {tmp_dir}")
            if os.path.exists(tmp_zip_file):
                os.remove(tmp_zip_file)
                logger.info(f"Cleaned up temp zip: {tmp_zip_file}")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


def search_artifacts(payload):
    """
    Handle POST /artifacts.

    All fields are optional:
      - name: regex (or simple substring if regex is invalid)
      - types: list of artifact types
      - version / version_range: version constraint string
      - page / page_num: 1-based page number
      - page_size / limit: number of items per page
    """
    name_query = (payload.get("name") or "").strip()
    types = payload.get("types", [])
    want_all_types = not types

    # support both "version" and "version_range"
    version_query = (
        payload.get("version") or payload.get("version_range") or ""
    ).strip()

    # pagination (1-based)
    page = payload.get("page") or payload.get("page_num") or 1
    page_size = payload.get("page_size") or payload.get("limit") or 50
    try:
        page = int(page)
    except (TypeError, ValueError):
        page = 1
    try:
        page_size = int(page_size)
    except (TypeError, ValueError):
        page_size = 50

    if page < 1:
        page = 1
    if page_size < 1:
        page_size = 1
    if page_size > 100:
        page_size = 100

    # table is tiny for this project, so a full scan is fine
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

    # stable ordering so pagination is predictable
    def _sort_key(it):
        return (
            str(it.get("model_name", "")),
            parse_semver(str(it.get("version", ""))) or (0, 0, 0),
        )

    items.sort(key=_sort_key)

    # slice out just the requested page
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    page_items = items[start_idx:end_idx]

    # autograder expects {"items": [...]} and not much else
    return make_response(200, {"items": page_items})


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
            {"error": "The lineage graph cannot be computed because the artifact metadata is missing or malformed."}
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
                    f"Parent model with repo_id '{parent_repo_id}' not found in registry for child '{item.get('repo_id')}'"
                )

    return make_response(200, {"nodes": nodes, "edges": edges})


def handler(event, context):
    method, path, body = parse_event(event)

    if not TABLE_NAME or not BUCKET_NAME:
        return make_response(500, {"error": "missing env vars for table/bucket"})

    # basic health check
    if method == "GET" and path == "/health":
        return make_response(200, {"status": "ok"})

    # tracks: spec allows this to just return an empty list for now
    if method == "GET" and path == "/tracks":
        return make_response(200, {"tracks": []})

    # authentication entry point
    if method == "PUT" and path == "/authenticate":
        if not USER_TABLE_NAME or not JWT_SECRET_KEY:
            return make_response(
                501, {"error": "This system does not support authentication."}
            )
        return authenticate_user(body)

    # everything else (except reset special-case below) is behind auth
    user_payload = get_validated_user(event)

    # reset route: needed by the autograder and for local testing
    if path == "/reset" and method in ("POST", "DELETE"):
        # for the class infrastructure, allow reset to work before we've
        # created any users so the default admin can be bootstrapped
        if not user_payload:
            try:
                out = reset_state()
                return make_response(200, out)
            except Exception as e:
                return make_response(500, {"error": str(e)})

        # if we *do* have a user, enforce admin-only
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

    if not user_payload:
        return make_response(
            403,
            {
                "error": "Authentication failed due to invalid or "
                "missing AuthenticationToken."
            },
        )

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

    # GET /artifact/{id}
    if method == "GET" and path.startswith("/artifact/") and path.count("/") == 2:
        art_id = path.split("/artifact/", 1)[-1]
        item = get_model_by_id(art_id)
        if not item:
            return make_response(404, {"error": "Model not found"})
        return make_response(200, item)
    
    # GET /artifact/model/{id}/lineage
    lineage_match = re.match(r"/artifact/model/([^/]+)/lineage", path)
    if method == "GET" and lineage_match:
        art_id = lineage_match.group(1)
        if not art_id:
            return make_response(400, {"error": "Missing artifact ID in path"})
        return get_lineage_graph(art_id)

    # POST /artifacts
    if method == "POST" and path == "/artifacts":
        try:
            return search_artifacts(body or {})
        except Exception as e:
            return make_response(400, {"error": str(e)})

    # anything else is a 404
    return make_response(404, {"error": f"Route not found: {method} {path}"})
