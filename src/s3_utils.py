import boto3
import os
import logging
from botocore.exceptions import ClientError

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize Boto3
s3_client = boto3.client('s3')
S3_BUCKET_NAME = os.environ.get('S3_BUCKET_NAME')

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
    Creates a test file, uploads it, downloads it, and verifies the operation.
    """
    import tempfile
    
    test_content = "This is a test file for S3 operations"
    test_key = "test-files/lambda-test.txt"
    
    # Create a temporary file for upload
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as tmp_file:
        tmp_file.write(test_content)
        upload_path = tmp_file.name
    
    try:
        # Test upload
        logger.info("Testing S3 upload...")
        upload_success = upload_model(upload_path, test_key)
        
        if not upload_success:
            return {"status": "FAILED", "error": "Upload failed"}
        
        # Test download
        logger.info("Testing S3 download...")
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.txt') as tmp_download:
            download_path = tmp_download.name
        
        downloaded_path = download_model(test_key, download_path)
        
        if not downloaded_path:
            return {"status": "FAILED", "error": "Download failed"}
        
        # Verify content
        with open(downloaded_path, 'r') as f:
            downloaded_content = f.read()
        
        if downloaded_content == test_content:
            return {
                "status": "SUCCESS", 
                "message": "S3 upload and download working correctly",
                "test_key": test_key,
                "bucket": S3_BUCKET_NAME
            }
        else:
            return {"status": "FAILED", "error": "Content mismatch after download"}
            
    except Exception as e:
        logger.error(f"Test failed with exception: {e}")
        return {"status": "FAILED", "error": str(e)}
    finally:
        # Clean up local files
        try:
            os.unlink(upload_path)
            if 'download_path' in locals():
                os.unlink(download_path)
        except:
            pass