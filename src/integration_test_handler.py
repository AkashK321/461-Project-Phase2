import logging
import os
import tempfile
import urllib.request
import hashlib
import zipfile

# Import both utility modules
import s3_utils
import db_utils

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def test_aws_integration(event, context):
    """
    This is the new Lambda handler for testing.
    It tests S3 upload/download AND DynamoDB save/get/delete.
    """
    logger.info("--- STARTING AWS INTEGRATION TEST ---")

    # Get bucket name from environment
    S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME")

    # --- 1. S3 TEST ---
    logger.info("--- Phase 1: Testing S3 Operations ---")

    model_files = {
        "config.json": "https://huggingface.co/google-bert/bert-base-uncased/resolve/main/config.json",
        "tokenizer_config.json": "https://huggingface.co/google-bert/bert-base-uncased/resolve/main/tokenizer_config.json",
    }
    s3_test_key = "test-models/integration-test-package.zip"
    created_item_id = None  # To store the DB item ID for cleanup

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            zip_path = os.path.join(temp_dir, "bert-package.zip")

            # Create zip package
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                for filename, url in model_files.items():
                    temp_file = os.path.join(temp_dir, filename)
                    urllib.request.urlretrieve(url, temp_file)
                    zipf.write(temp_file, filename)

            with open(zip_path, "rb") as f:
                original_hash = hashlib.md5(f.read()).hexdigest()

            # Test S3 Upload
            logger.info("Testing S3 upload...")
            if not s3_utils.upload_model(zip_path, s3_test_key):
                raise Exception("s3_utils.upload_model failed")

            # Test S3 Download
            logger.info("Testing S3 download...")
            download_path = os.path.join(temp_dir, "downloaded-package.zip")
            if not s3_utils.download_model(s3_test_key, download_path):
                raise Exception("s3_utils.download_model failed")

            # Verify file integrity
            with open(download_path, "rb") as f:
                downloaded_hash = hashlib.md5(f.read()).hexdigest()

            if original_hash != downloaded_hash:
                raise Exception(f"File integrity check failed. Hashes do not match.")

            logger.info("S3 operations test SUCCEEDED.")

            # --- 2. DYNAMODB TEST ---
            logger.info("--- Phase 2: Testing DynamoDB Operations ---")

            dummy_scores = {"net_score": 0.99, "license": 1.0}

            # 2a. Save metadata
            logger.info(f"Saving metadata for S3 key: {s3_test_key}")
            created_item = db_utils.save_model_metadata(
                "test-bert-model", "1.0.0", s3_test_key, dummy_scores
            )

            if not created_item:
                raise Exception("db_utils.save_model_metadata failed, returned None.")

            created_item_id = created_item.get("id")
            logger.info(f"Save successful. New item ID: {created_item_id}")

            # 2b. Get metadata
            logger.info(f"Getting metadata for ID: {created_item_id}")
            retrieved_item = db_utils.get_model_by_id(created_item_id)

            if not retrieved_item:
                raise Exception(
                    f"db_utils.get_model_by_id failed for ID: {created_item_id}"
                )

            # 2c. Verify metadata
            if retrieved_item.get("s3_key") != s3_test_key:
                raise Exception("Data mismatch: s3_key does not match.")

            logger.info("DynamoDB Get successful. Data verification passed.")

            # --- 3. COMBINED SUCCESS ---
            logger.info("--- AWS INTEGRATION TEST SUCCEEDED ---")
            return {
                "status": "SUCCESS",
                "message": "S3 and DynamoDB operations are working correctly.",
                "s3_key": s3_test_key,
                "dynamodb_id": created_item_id,
            }

    except Exception as e:
        logger.error(f"Test FAILED: {e}")
        return {"status": "FAILED", "error": str(e)}

    finally:
        # --- 4. CLEANUP ---
        if created_item_id:
            logger.info(f"Cleaning up DynamoDB record: {created_item_id}")
            db_utils.delete_model_metadata(created_item_id)

        logger.info(f"Cleaning up S3 object: {s3_test_key}")
        s3_utils.delete_model(s3_test_key)
