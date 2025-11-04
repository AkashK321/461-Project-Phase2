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


def delete_model(s3_object_key):
    """
    Deletes a model file (object) from the S3 bucket.
    """
    if not S3_BUCKET_NAME:
        logger.error("S3_BUCKET_NAME environment variable not set.")
        return False

    try:
        s3_client.delete_object(Bucket=S3_BUCKET_NAME, Key=s3_object_key)
        logger.info(f"Successfully deleted {s3_object_key} from {S3_BUCKET_NAME}")
        return True
    except ClientError as e:
        logger.error(f"Failed to delete {s3_object_key}: {e}")
        return False
