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
from aws_modules.db_utils import save_model_metadata, get_model_by_id
from utils.lineage_utils import get_base_model_from_card
from scorer.metrics.base import get_repo_id
from scorer.url_handler.base import classify_url
from aws_modules.api_utils import make_response

from aws_modules.auth import (
    authenticate_user,
    get_validated_user,
    register_user,
    hash_password,
)

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

# wire up AWS stuff once
dynamodb = boto3.resource("dynamodb")
s3 = boto3.client("s3")

TABLE_NAME = os.getenv("DYNAMODB_TABLE_NAME", "")
BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "")
USER_TABLE_NAME = os.getenv("USER_DYNAMODB_TABLE_NAME", "")
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "a-very-unsafe-default-secret")


def parse_event(event):
    # extracts method, path, body from API Gateway event
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
    # wipe DynamoDB table
    tbl = dynamodb.Table(TABLE_NAME)
    scan = tbl.scan(ProjectionExpression="#i", ExpressionAttributeNames={"#i": "id"})
    ids = [it["id"] for it in scan.get("Items", [])]
    if ids:
        with tbl.batch_writer() as batch:
            for _id in ids:
                batch.delete_item(Key={"id": _id})

    # nuke S3 objects under models/
    if BUCKET_NAME:
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix="models/"):
            objs = [{"Key": o["Key"]} for o in page.get("Contents", [])]
            if objs:
                s3.delete_objects(Bucket=BUCKET_NAME, Delete={"Objects": objs})

    if USER_TABLE_NAME:
        user_tbl = dynamodb.Table(USER_TABLE_NAME)
        # Credentials from spec
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
    Ingests an artifact (model) via POST /artifact/{type}
    Expects payload with:
      - urls (list of strings): at least 1 url pointing to the model
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

    # TODO: Per the spec , you must add logic here to:
    # 1. Calculate all non-latency metrics for the model *before* downloading.
    # 2. Check if all scores are >= 0.5.
    # 3. If not, return an error and do not proceed with ingestion.
    logger.warning("Metric pre-check not implemented. Proceeding directly to download.")

    tmp_dir = f"/tmp/{str(uuid.uuid4())}"
    tmp_zip_file = ""
    base_model_repo = get_base_model_from_card(repo)
    logger.info(f"Base model repo from card: {base_model_repo}")

    try:
        # Download all files from the Hugging Face repo
        logger.info(f"Downloading model '{repo}' to '{tmp_dir}'")
        snapshot_download(
            repo_id=repo,
            local_dir=tmp_dir,
        )

        # Zip the downloaded directory
        zip_name = f"{name}"
        tmp_zip_path_base = f"/tmp/{zip_name}"
        # Creates a zip file (e.g., /tmp/bert-base-uncased.zip)
        tmp_zip_file = shutil.make_archive(tmp_zip_path_base, "zip", tmp_dir)
        final_zip_name = f"{name}.zip"

        # --- Upload and Save to DB ---
        model_id = str(uuid.uuid4())
        s3_key = f"models/{model_id}/{final_zip_name}"

        # Upload to S3
        logger.info(f"Uploading '{tmp_zip_file}' to S3 key '{s3_key}'")
        ok = upload_model(tmp_zip_file, s3_key)
        if ok is False:
            return make_response(500, {"error": "S3 upload failed"})

        version = "v1"  # TODO: You might want to parse this from HF
        scores = {}  # TODO: Add scores from the pre-check
        created_at = datetime.now(timezone.utc).isoformat()

        # Save metadata to DynamoDB
        item = save_model_metadata(name, version, s3_key, scores)
        if not item:
            return make_response(500, {"error": "failed to store metadata"})

        # Add additional metadata
        tbl = dynamodb.Table(TABLE_NAME)
        tbl.update_item(
            Key={"id": item["id"]},
            UpdateExpression="SET #t = :t, #c = :c, #fn = :fn, \
                #url = :url, #rid = :rid, #brid = :brid",
            ExpressionAttributeNames={
                "#t": "type",
                "#c": "created_at",
                "#fn": "filename",
                "#url": "source_url",
                "#rid": "repo_id",
                "#brid": "base_model_repo_id",
            },
            ExpressionAttributeValues={
                ":t": art_type,
                ":c": created_at,
                ":fn": final_zip_name,
                ":url": url,
                ":rid": repo,
                ":brid": base_model_repo,
            },
        )

        logger.info(f"Successfully ingested model {model_id} from {url}")
        return make_response(201, {"id": item["id"], "s3_key": s3_key, "model": name})
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        return make_response(500, {"error": f"Internal server error: {str(e)}"})
    finally:
        # Cleanup temp files and directories
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
    Searches/list artifacts via POST /artifacts
    Expects payload with optional fields:
      - name (string or regex)
      - types (list of strings)
    """
    name_query = (payload.get("name") or "").strip()
    types = payload.get("types", [])
    want_all_types = not types

    tbl = dynamodb.Table(TABLE_NAME)
    scan = tbl.scan()
    items = scan.get("Items", [])

    # filter by type (if provided)
    if not want_all_types:
        items = [
            it
            for it in items
            if str(it.get("type", "")).lower() in {t.lower() for t in types}
        ]

    # filter by name (regex-like); if empty, return all from above
    if name_query:
        try:
            rx = re.compile(name_query)
            items = [it for it in items if rx.search(str(it.get("model_name", "")))]
        except re.error:
            # if given a bad regex, fallback to contains
            items = [it for it in items if name_query in str(it.get("model_name", ""))]

    return make_response(200, {"items": items})


def handler(event, context):
    method, path, body = parse_event(event)

    if not TABLE_NAME or not BUCKET_NAME:
        return make_response(500, {"error": "missing env vars for table/bucket"})

    # ---- health ----
    if method == "GET" and path == "/health":
        return make_response(200, {"status": "ok"})

    # ---- tracks (baseline: no auth, empty is fine) ----
    if method == "GET" and path == "/tracks":
        return make_response(200, {"tracks": []})

    if method == "PUT" and path == "/authenticate":
        if not USER_TABLE_NAME or not JWT_SECRET_KEY:
            return make_response(
                501, {"error": "This system does not support authentication."}
            )
        return authenticate_user(body)

    # --- Protected Routes ---
    user_payload = get_validated_user(event)

    # ---- reset (autograder gates on this) ----
    if path == "/reset" and method in ("POST", "DELETE"):
        if not user_payload:
            # 403 for auth failure
            return make_response(
                403,
                {
                    "error": "Authentication failed due to invalid or "
                    "missing AuthenticationToken."
                },
            )

        user_roles = user_payload.get("roles", [])

        if "admin" not in user_roles:
            # 401 for permission denied
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

    # ---- ingest: POST /artifact/{type} ----
    if method == "POST" and path.startswith("/artifact/") and path.count("/") == 2:
        # /artifact/<type>
        art_type = path.split("/artifact/", 1)[-1]
        try:
            return ingest_artifact(art_type, body)
        except KeyError as ke:
            return make_response(400, {"error": f"missing field: {str(ke)}"})
        except Exception as e:
            return make_response(400, {"error": str(e)})

    # ---- read-one: GET /artifact/{id} ----
    if method == "GET" and path.startswith("/artifact/") and path.count("/") == 2:
        # /artifact/<id>
        art_id = path.split("/artifact/", 1)[-1]
        item = get_model_by_id(art_id)
        if not item:
            return make_response(404, {"error": "Model not found"})
        return make_response(200, item)

    # ---- search/list: POST /artifacts ----
    if method == "POST" and path == "/artifacts":
        try:
            return search_artifacts(body or {})
        except Exception as e:
            return make_response(400, {"error": str(e)})

    # default
    return make_response(404, {"error": f"Route not found: {method} {path}"})
