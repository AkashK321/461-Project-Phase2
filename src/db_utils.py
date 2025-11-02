import boto3
import os
import logging
import uuid

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize Boto3
dynamodb = boto3.resource("dynamodb")
TABLE_NAME = os.environ.get("DYNAMODB_TABLE_NAME")

if not TABLE_NAME:
    logger.warning("DYNAMODB_TABLE_NAME environment variable is not set!")
    table = None
else:
    table = dynamodb.Table(TABLE_NAME)


def save_model_metadata(name, version, s3_key, scores):
    """
    Saves a new model's metadata to the DynamoDB table.
    """
    if not table:
        logger.error("DynamoDB table is not initialized. Check env var.")
        return None

    try:
        new_id = str(uuid.uuid4())
        item = {
            "id": new_id,  # Primary key
            "model_name": name,
            "version": version,
            "s3_key": s3_key,  # Link to the S3 file
            "scores": scores,  # Your Phase 1 & 2 scores
        }

        table.put_item(Item=item)
        logger.info(f"Successfully saved metadata for {name} (ID: {new_id})")
        return item

    except Exception as e:
        logger.error(f"Failed to save metadata: {e}")
        return None


def get_model_by_id(model_id):
    """
    Fetches a single model's metadata by its ID.
    """
    if not table:
        logger.error("DynamoDB table is not initialized. Check env var.")
        return None

    try:
        response = table.get_item(Key={"id": model_id})
        return response.get("Item")
    except Exception as e:
        logger.error(f"Failed to get item {model_id}: {e}")
        return None


def delete_model_metadata(model_id):
    """
    Deletes a model's metadata record by its ID.
    (Used for test cleanup)
    """
    if not table:
        logger.error("DynamoDB table is not initialized. Check env var.")
        return False

    try:
        table.delete_item(Key={"id": model_id})
        logger.info(f"Successfully deleted item {model_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to delete item {model_id}: {e}")
        return False
