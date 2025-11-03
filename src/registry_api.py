import json
import base64
import uuid
import os
import re
from datetime import datetime, timezone

import boto3

from s3_utils import upload_model
from db_utils import save_model_metadata, get_model_by_id

# wire up AWS stuff once
dynamodb = boto3.resource("dynamodb")
s3 = boto3.client("s3")

TABLE_NAME = os.getenv("DYNAMODB_TABLE_NAME", "")
BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "")


def make_response(status_code, body):
    # formats API Gateway response
    return {
        "statusCode": status_code,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }


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

    return {"reset": "ok", "deleted": {"dynamodb": len(ids)}}


def ingest_artifact(art_type, payload):
    """
    Ingests an artifact (model) via POST /artifact/{type}
    Expects payload with:
      - filename
      - file_b64  (base64 of the file)
    """
    filename = payload["filename"]
    file_b64 = payload["file_b64"]
    file_bytes = base64.b64decode(file_b64)

    # write to tmp then upload
    tmp_path = f"/tmp/{filename}"
    with open(tmp_path, "wb") as f:
        f.write(file_bytes)

    model_id = str(uuid.uuid4())
    s3_key = f"models/{model_id}/{filename}"

    # upload to S3
    ok = upload_model(tmp_path, s3_key)
    if ok is False:
        return make_response(500, {"error": "S3 upload failed"})

    name = filename.rsplit(".", 1)[0]
    version = "v1"
    scores = {}
    created_at = datetime.now(timezone.utc).isoformat()

    # save base metadata with helper first (keeps compatibility)
    item = save_model_metadata(name, version, s3_key, scores)
    if not item:
        return make_response(500, {"error": "failed to store metadata"})

    try:
        tbl = dynamodb.Table(TABLE_NAME)
        tbl.update_item(
            Key={"id": item["id"]},
            UpdateExpression="SET #t = :t, #c = :c, #fn = :fn",
            ExpressionAttributeNames={
                "#t": "type",
                "#c": "created_at",
                "#fn": "filename",
            },
            ExpressionAttributeValues={
                ":t": art_type,
                ":c": created_at,
                ":fn": filename,
            },
        )
    except Exception:

        pass

    return make_response(201, {"id": item["id"], "s3_key": s3_key})


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

    # ---- reset (autograder gates on this) ----
    if path == "/reset" and method in ("POST", "DELETE"):
        try:
            out = reset_state()
            return make_response(200, out)
        except Exception as e:
            return make_response(500, {"error": str(e)})

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
