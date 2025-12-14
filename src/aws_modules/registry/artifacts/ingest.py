import os
import json
import shutil
import uuid
import requests
import zipfile
import hashlib
from datetime import datetime, timezone

from huggingface_hub import snapshot_download
from huggingface_hub.utils import GatedRepoError

from aws_modules.s3_utils import upload_model
from aws_modules.db_utils import save_model_metadata
from utils.lineage_utils import get_base_model_from_card
from scorer.metrics.base import get_repo_id
from aws_modules.api_utils import make_response

from aws_modules.registry.system import logger
from aws_modules.registry.context import (
    dynamodb,
    lambda_client,
    TABLE_NAME,
    SCORER_FUNCTION_NAME,
    FEATURE_FLAG_FORCE_INGESTION,
)


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def get_readme_content(directory):
    """
    Scans the given directory for a README file and returns its content.
    Limits content to 10KB to avoid DynamoDB size limits.
    """
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().startswith("readme"):
                try:
                    with open(os.path.join(root, file), "r", errors="ignore") as f:
                        return f.read(10000)
                except Exception as e:
                    logger.warning(f"Failed to read README file {file}: {e}")
    return ""


def ingest_artifact(artifact_type, payload):
    """
    Ingests an artifact. Supports 'model', 'dataset', 'code'.
    Assumes payload['url'] is a valid HTTPS URL.
    """
    logger.info(f"--- Starting Ingestion (Type: {artifact_type}) ---")

    logger.info(f"Raw Payload: {json.dumps(payload)}")

    try:
        url = payload.get("url")

        logger.info(f"Received URL: {url}")

        if not url or not isinstance(url, str):
            return make_response(
                400, {"error": "payload must have a non-empty 'url' string"}
            )

        urls = [url]

        # 1. Classification & Type override
        if "github.com" in url:
            if artifact_type != "code" and artifact_type != "dataset":
                logger.info(
                    f"Detected GitHub URL. Switching type from {artifact_type} to code."
                )
                artifact_type = "code"
        elif "huggingface.co" in url:
            if "/datasets/" in url and artifact_type != "dataset":
                artifact_type = "dataset"

        # 2. Extract Name/Repo
        if "github.com" in url:
            parts = url.split("github.com/")[-1].strip("/").split("/")
            repo = f"{parts[0]}/{parts[1]}" if len(parts) >= 2 else parts[0]
            if repo.endswith(".git"):
                repo = repo[:-4]
        else:
            repo = get_repo_id(url, artifact_type) or url.split("/")[-1]

        # Use the name provided in the payload
        name = payload.get("name", "").strip()
        if not name:
            # Fallback to extraction from repo/url if name is not provided
            name = repo.split("/")[-1] if "/" in repo else repo

        logger.info(f"Resolved Name: {name}, Repo: {repo}")

    except Exception as e:
        logger.error(f"URL parsing error for payload '{payload}': {e}")
        return make_response(400, {"error": "unable to parse url"})

    # --- Invoke Scorer ---
    scorer_function_name = SCORER_FUNCTION_NAME
    scores = {}
    if scorer_function_name:
        try:
            scorer_payload = json.dumps({"urls": urls})
            response = lambda_client.invoke(
                FunctionName=scorer_function_name,
                InvocationType="RequestResponse",
                Payload=scorer_payload,
            )
            raw_response = response["Payload"].read().decode()
            response_payload = json.loads(raw_response)
            if response_payload.get("statusCode") == 200:
                scores_list = json.loads(response_payload["body"])
                if scores_list:
                    scores = scores_list[0]
        except Exception as e:
            logger.error(f"Scorer error (likely timeout): {e}")

    # --- Quality Gate ---
    metrics_map = {
        "code": ["code_quality"],
        "dataset": ["dataset_quality", "dataset_and_code_score"],
        "model": [
            "size_score",
            "performance_claims",
            "ramp_up_time",
            "license",
        ],
    }
    required_metrics = metrics_map.get(artifact_type, [])
    failing_metrics = []

    for metric in required_metrics:
        if metric in scores:
            val = scores[metric]
            if isinstance(val, dict):
                val = sum(val.values()) / len(val) if val else 0

            if val < 0.5:
                failing_metrics.append(f"{metric}: {val}")
        else:
            pass

    if failing_metrics and not FEATURE_FLAG_FORCE_INGESTION:
        logger.warning(f"Rejecting {url} due to metrics: {failing_metrics}")
        return make_response(
            424, {"error": "Insufficient quality metrics", "failing": failing_metrics}
        )

    # --- Download & Package ---
    tmp_dir = f"/tmp/{str(uuid.uuid4())}"
    tmp_zip_file = ""
    tmp_download_path = ""

    base_model_repo = None
    if artifact_type == "model":
        base_model_repo, _, _ = get_base_model_from_card(repo)

    try:
        os.makedirs(tmp_dir, exist_ok=True)

        try:
            os.chmod(tmp_dir, 0o700)
        except Exception:
            pass

        logger.info(f"Downloading {artifact_type} '{repo}' to '{tmp_dir}'")

        if "github.com" in url:
            # 1. Determine Download URL (Default Branch)
            zip_url = f"https://github.com/{repo}/archive/refs/heads/main.zip"
            logger.info(f"Attempting download from: {zip_url}")

            try:
                r_head = requests.head(zip_url, timeout=5)
                if r_head.status_code != 200:
                    zip_url = f"https://github.com/{repo}/archive/refs/heads/master.zip"
                    logger.info(f"Main failed, trying master: {zip_url}")
            except Exception:
                pass

            # 2. Stream Download
            tmp_download_path = os.path.join(tmp_dir, "downloaded_repo.zip")

            with requests.get(zip_url, stream=True) as r_zip:
                if r_zip.status_code != 200:
                    logger.error(
                        f"Failed to download zip from {zip_url}. \
                        Status: {r_zip.status_code}"
                    )
                    raise Exception(
                        f"Failed to download repo zip: HTTP {r_zip.status_code}"
                    )

                with open(tmp_download_path, "wb") as f:
                    for chunk in r_zip.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

            # 2b. Record integrity hash immediately after download
            try:
                os.chmod(tmp_download_path, 0o600)
            except Exception:
                pass

            expected_zip_sha256 = _sha256_file(tmp_download_path)
            logger.info(f"Downloaded repo zip SHA256: {expected_zip_sha256}")

            # 3. Extract with Filtering
            excluded_extensions = {
                # Media
                ".png",
                ".jpg",
                ".jpeg",
                ".gif",
                ".bmp",
                ".tiff",
                ".svg",
                ".webp",
                ".mp4",
                ".mov",
                ".avi",
                ".mp3",
                ".wav",
                # Archives
                ".zip",
                ".tar",
                ".gz",
                ".7z",
                ".rar",
                # Documents
                ".pdf",
                ".doc",
                ".docx",
                ".ppt",
                ".pptx",
                ".xls",
                ".xlsx",
                # Compiled/Binary
                ".pyc",
                ".pyo",
                ".pyd",
                ".o",
                ".obj",
                ".dll",
                ".exe",
                ".so",
                ".dylib",
                ".class",
                ".jar",
                # Large Data
                ".parquet",
                ".arrow",
                ".csv",
                ".tsv",
                ".jsonl",
                ".h5",
                ".bin",
                ".safetensors",
                # Other
                ".DS_Store",
            }
            excluded_dirs = {
                "__pycache__",
                ".git",
                ".idea",
                ".vscode",
                ".venv",
                "venv",
                "env",
                "node_modules",
            }

            def is_allowed(filename):
                parts = filename.split("/")
                for part in parts:
                    if part in excluded_dirs:
                        return False
                _, ext = os.path.splitext(filename)
                if ext.lower() in excluded_extensions:
                    return False
                return True

            try:

                # 3a. Verify integrity hash again immediately before extraction
                verify_zip_sha256 = _sha256_file(tmp_download_path)
                if verify_zip_sha256 != expected_zip_sha256:
                    raise Exception("Repo zip failed SHA256 check before extraction")

                with zipfile.ZipFile(tmp_download_path, "r") as z:
                    for file_info in z.infolist():
                        if not file_info.filename.endswith("/"):
                            if is_allowed(file_info.filename):
                                z.extract(file_info, tmp_dir)
            except zipfile.BadZipFile:
                raise Exception("Downloaded file is not a valid zip file")

            items = os.listdir(tmp_dir)
            items = [i for i in items if i != "downloaded_repo.zip"]

            if len(items) == 1 and os.path.isdir(os.path.join(tmp_dir, items[0])):
                top_level = os.path.join(tmp_dir, items[0])
                for item in os.listdir(top_level):
                    shutil.move(os.path.join(top_level, item), tmp_dir)
                os.rmdir(top_level)

        else:
            # --- Use HF Hub ---
            allow_patterns = ["*.json", "*.md", "*.txt", "*.py", "*.yaml", "*.yml"]

            try:
                snapshot_download(
                    repo_id=repo,
                    local_dir=tmp_dir,
                    repo_type=artifact_type,
                    allow_patterns=allow_patterns,
                )
            except (GatedRepoError, requests.HTTPError) as e:
                logger.warning(f"Auth error downloading {repo}: {e}")
                return make_response(424, {"error": "Artifact requires authentication"})
            except Exception as e:
                msg = str(e)
                if "401" in msg or "403" in msg:
                    return make_response(
                        424, {"error": "Artifact requires authentication"}
                    )
                raise e

        # --- Capture README content for searching ---
        readme_content = get_readme_content(tmp_dir)

        # Create Zip
        zip_name = f"{name}"
        tmp_zip_path_base = f"/tmp/{zip_name}"
        tmp_zip_file = shutil.make_archive(tmp_zip_path_base, "zip", tmp_dir)
        final_zip_name = f"{name}.zip"

        # Upload to S3
        model_id = str(uuid.uuid4())
        s3_key = f"artifacts/{artifact_type}/{model_id}/{final_zip_name}"

        if upload_model(tmp_zip_file, s3_key) is False:
            return make_response(500, {"error": "S3 upload failed"})

        # Save Metadata
        version = "1.0.0"
        created_at = datetime.now(timezone.utc).isoformat()

        item = save_model_metadata(name, version, s3_key, scores, artifact_type)
        if not item:
            return make_response(500, {"error": "failed to store metadata"})

        tbl = dynamodb.Table(TABLE_NAME)

        tbl.update_item(
            Key={"id": item["id"]},
            UpdateExpression="SET #c = :c, #fn = :fn, \
            #url = :url, #rid = :rid, #bm = :bm, #readme = :readme",
            ExpressionAttributeNames={
                "#c": "created_at",
                "#fn": "filename",
                "#url": "source_url",
                "#rid": "repo_id",
                "#bm": "base_model_repo_id",
                "#readme": "readme",
            },
            ExpressionAttributeValues={
                ":c": created_at,
                ":fn": final_zip_name,
                ":url": url,
                ":rid": repo,
                ":bm": base_model_repo,
                ":readme": readme_content,
            },
        )

        return make_response(
            201,
            {
                "metadata": {
                    "name": name,
                    "id": item["id"],
                    "type": artifact_type,
                    "version": version,
                },
                "data": {"url": url},
            },
        )

    except Exception as e:
        logger.error(f"Ingestion failed for URL '{url}': {e}")
        return make_response(500, {"error": f"Internal error: {str(e)}"})
    finally:
        if tmp_download_path and os.path.exists(tmp_download_path):
            try:
                os.remove(tmp_download_path)
            except Exception:
                pass
        if os.path.exists(tmp_dir):
            try:
                shutil.rmtree(tmp_dir)
            except Exception:
                pass
        if tmp_zip_file and os.path.exists(tmp_zip_file):
            try:
                os.remove(tmp_zip_file)
            except Exception:
                pass
