import os
import json
import base64
import uuid
from datetime import datetime, timezone

from src.s3_utils import upload_model
from src.db_utils import put_model_metadata, get_model_metadata

S3_BUCKET = os.environ.get("S3_BUCKET_NAME")
DDB_TABLE = os.environ.get("DDB_TABLE_NAME")

def make_response(status_code, body):
    """Formats HTTP response for API Gateway."""
    return {
        "statusCode": status_code,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }

def handler(event, context):
    """Main Lambda handler for registry API routes."""
    path = event.get("rawPath", "")
    method = event.get("requestContext", {}).get("http", {}).get("method", "GET")

    # Handle POST /upload
    if method == "POST" and path == "/upload":
        try:
            body = event.get("body") or "{}"
            if event.get("isBase64Encoded"):
                body = base64.b64decode(body).decode("utf-8")
            data = json.loads(body)

            filename = data["filename"]
            file_b64 = data["file_b64"]
            file_bytes = base64.b64decode(file_b64)

            tmp_path = f"/tmp/{filename}"
            with open(tmp_path, "wb") as f:
                f.write(file_bytes)

            model_id = str(uuid.uuid4())
            s3_key = f"models/{model_id}/{filename}"

            # Upload file to S3
            if not upload_model(tmp_path, s3_key):
                return make_response(500, {"error": "S3 upload failed"})

            # Store model metadata in DynamoDB
            item = {
                "id": model_id,
                "name": filename,
                "s3_key": s3_key,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "scores": {},
            }

            if not put_model_metadata(DDB_TABLE, item):
                return make_response(500, {"error": "Failed to store metadata"})

            return make_response(201, {"id": model_id, "s3_key": s3_key})

        except Exception as e:
            return make_response(400, {"error": str(e)})

    # Handle GET /artifact/{id}
    if method == "GET" and path.startswith("/artifact/"):
        model_id = path.split("/artifact/", 1)[-1]
        item = get_model_metadata(DDB_TABLE, model_id)

        if not item:
            return make_response(404, {"error": "Model not found"})

        return make_response(200, item)

    # Unrecognized route
    return make_response(404, {"error": f"Route not found: {method} {path}"})
