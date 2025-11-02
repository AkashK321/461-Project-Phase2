import boto3
import os
import logging
from botocore.exceptions import ClientError

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize Boto3
s3_client = boto3.client("s3")
S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME")


def upload_model(local_file_path, s3_object_key):
    """
    Uploads a model file from a local path to S3.
    """
    if not S3_BUCKET_NAME:
        logger.error("S3_BUCKET_NAME environment variable not set.")
        return False

    try:
        s3_client.upload_file(local_file_path, S3_BUCKET_NAME, s3_object_key)
        logger.info(f"Successfully uploaded {s3_object_key}")
        return True
    except ClientError as e:
        logger.error(f"Failed to upload {local_file_path}: {e}")
        return False


def download_model(s3_object_key, local_download_path):
    """
    Downloads a model file from S3 to a local path.
    """
    if not S3_BUCKET_NAME:
        logger.error("S3_BUCKET_NAME environment variable not set.")
        return None

    try:
        s3_client.download_file(S3_BUCKET_NAME, s3_object_key, local_download_path)
        logger.info(f"Successfully downloaded {s3_object_key}")
        return local_download_path
    except ClientError as e:
        logger.error(f"Failed to download {s3_object_key}: {e}")
        return None


def test_s3_operations():
    """
    Test function for Lambda console to verify S3 upload/download functionality.
    Downloads real model files from HuggingFace, creates a zip package,
    and tests upload/download using the upload_model/download_model functions.
    """
    import tempfile
    import urllib.request
    import hashlib
    import zipfile

    # Download a couple small files from BERT model to create a zip
    model_files = {
        "config.json": "https://huggingface.co/google-bert/bert-base-uncased/"
        "resolve/main/config.json",
        "tokenizer_config.json": "https://huggingface.co/google-bert/bert-base-uncased/"
        "resolve/main/tokenizer_config.json",
    }

    test_key = "test-models/bert-package.zip"

    try:
        # Create temporary directory for model files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Download model files
            logger.info("Downloading model files from HuggingFace...")

            # Create zip package
            zip_path = os.path.join(temp_dir, "bert-package.zip")

            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                for filename, url in model_files.items():
                    try:
                        # Download file to temp location
                        temp_file = os.path.join(temp_dir, filename)
                        urllib.request.urlretrieve(url, temp_file)
                        # Add to zip
                        zipf.write(temp_file, filename)
                        logger.info(f"Added {filename} to zip")
                    except Exception as e:
                        logger.warning(f"Failed to add {filename}: {e}")

            # Verify zip was created
            if not os.path.exists(zip_path):
                return {"status": "FAILED", "error": "Failed to create zip package"}

            # Get original zip hash for verification
            with open(zip_path, "rb") as f:
                original_hash = hashlib.md5(f.read()).hexdigest()

            # Test upload using existing upload_model function
            logger.info("Testing S3 upload with upload_model function...")
            upload_success = upload_model(zip_path, test_key)

            if not upload_success:
                return {"status": "FAILED", "error": "Upload of zip package failed"}

            # Test download using existing download_model function
            logger.info("Testing S3 download with download_model function...")
            download_path = os.path.join(temp_dir, "downloaded-package.zip")
            downloaded_path = download_model(test_key, download_path)

            if not downloaded_path:
                return {"status": "FAILED", "error": "Download of zip package failed"}

            # Verify file integrity
            with open(downloaded_path, "rb") as f:
                downloaded_hash = hashlib.md5(f.read()).hexdigest()
            downloaded_size = os.path.getsize(downloaded_path)

            # Verify zip contents
            try:
                with zipfile.ZipFile(downloaded_path, "r") as zipf:
                    zip_contents = zipf.namelist()
            except Exception as e:
                return {
                    "status": "FAILED",
                    "error": f"Downloaded zip file is corrupted: {e}",
                }

            if original_hash == downloaded_hash:
                return {
                    "status": "SUCCESS",
                    "message": "Upload_model/download_model functions work!",
                    "test_key": test_key,
                    "bucket": S3_BUCKET_NAME,
                    "zip_size_bytes": downloaded_size,
                    "zip_contents": zip_contents,
                    "md5_hash": downloaded_hash,
                }
            else:
                return {
                    "status": "FAILED",
                    "error": "Zip file integrity check failed",
                    "original_hash": original_hash,
                    "downloaded_hash": downloaded_hash,
                }

    except Exception as e:
        logger.error(f"Test failed with exception: {e}")
        return {"status": "FAILED", "error": str(e)}
