import boto3
from boto3.dynamodb.conditions import Attr
import os
import logging
import uuid
from decimal import Decimal

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


def floats_to_decimals(obj):
    """
    Recursively converts all float values in a dictionary or list to Decimal.
    This is necessary for DynamoDB, which does not support floats.
    """
    if isinstance(obj, list):
        return [floats_to_decimals(i) for i in obj]
    elif isinstance(obj, dict):
        return {k: floats_to_decimals(v) for k, v in obj.items()}
    elif isinstance(obj, float):
        # Convert float to string, then to Decimal for precision
        return Decimal(str(obj))
    else:
        return obj


def save_model_metadata(name, version, s3_key, scores):
    """
    Saves a new model's metadata to the DynamoDB table.
    """
    if not table:
        logger.error("DynamoDB table is not initialized. Check env var.")
        return None

    try:
        new_id = str(uuid.uuid4())

        scores_with_decimal = floats_to_decimals(scores)

        item = {
            "id": new_id,  # Primary key
            "model_name": name,
            "version": version,
            "s3_key": s3_key,
            "scores": scores_with_decimal,
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
    
def get_model_by_repo_id(repo_id):
    """
    Finds a model in DynamoDB by its 'repo_id' using a scan.
    Note: A scan is inefficient on large tables. A GSI would be better.
    """
    try:
        tbl = dynamodb.Table(TABLE_NAME)
        response = tbl.scan(FilterExpression=Attr("repo_id").eq(repo_id))
        items = response.get("Items", [])
        if items:
            return items[0]  # Assume repo_id is unique
        return None
    except Exception as e:
        logger.error(f"Error scanning for repo_id '{repo_id}': {e}")
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
