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


def generate_presigned_download_url(s3_object_key, expiration=3600):
    """
    Generates a presigned URL to download a file from S3.
    
    :param s3_object_key: The key of the object in S3.
    :param expiration: Time in seconds for the presigned URL to remain valid.
    :return: The presigned URL as a string, or None if an error occurred.
    """
    if not S3_BUCKET_NAME:
        logger.error("S3_BUCKET_NAME environment variable not set.")
        return None

    try:
        response = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': S3_BUCKET_NAME, 'Key': s3_object_key},
            ExpiresIn=expiration
        )
        logger.info(f"Generated presigned URL for {s3_object_key}")
        return response
    except ClientError as e:
        logger.error(f"Failed to generate presigned URL for {s3_object_key}: {e}")
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
