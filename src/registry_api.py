import json
import base64
import uuid

from s3_utils import upload_model
from db_utils import save_model_metadata, get_model_by_id

def make_response(status_code, body):
    """Return HTTP-style JSON response."""
    return {
        "statusCode": status_code,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }

def handler(event, context):
    """Lambda handler for model registry API routes."""
    path = event.get("rawPath", "")
    method = event.get("requestContext", {}).get("http", {}).get("method", "GET")

    # POST /upload
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

            # upload to S3 (helper raises on error; returns False only if it handled failure)
            result = upload_model(tmp_path, s3_key)
            if result is False:
                return make_response(500, {"error": "S3 upload failed"})

            name = filename.rsplit(".", 1)[0]
            version = "v1"
            scores = {}

            item = save_model_metadata(name, version, s3_key, scores)
            if not item:
                return make_response(500, {"error": "failed to store metadata"})

            return make_response(201, {"id": item["id"], "s3_key": s3_key})

        except Exception as e:
            return make_response(400, {"error": str(e)})

    # GET /artifact/{id}
    if method == "GET" and path.startswith("/artifact/"):
        model_id = path.split("/artifact/", 1)[-1]
        item = get_model_by_id(model_id)

        if not item:
            return make_response(404, {"error": "Model not found"})

        return make_response(200, item)

    return make_response(404, {"error": f"Route not found: {method} {path}"})

