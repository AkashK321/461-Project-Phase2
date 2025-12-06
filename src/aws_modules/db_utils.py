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


def save_model_metadata(name, version, s3_key, scores, artifact_type="model"):
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
            "type": artifact_type,
        }

        table.put_item(Item=item)
        logger.info(f"Successfully saved metadata for {name} (ID: {new_id})")
        return item

    except Exception as e:
        logger.error(f"Failed to save metadata: {e}")
        return None


def update_model_scores(model_id, new_scores):
    """
    Updates the 'scores' attribute for a given model ID.
    """
    if not table:
        logger.error("DynamoDB table is not initialized. Check env var.")
        return False

    try:
        scores_with_decimal = floats_to_decimals(new_scores)

        table.update_item(
            Key={"id": model_id},
            UpdateExpression="SET #s = :s",
            ExpressionAttributeNames={"#s": "scores"},
            ExpressionAttributeValues={":s": scores_with_decimal},
        )
        logger.info(f"Successfully updated scores for model ID: {model_id}")
        return True

    except Exception as e:
        logger.error(f"Failed to update scores for model {model_id}: {e}")
        return False


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


def get_model_by_model_name(name, dynamodb_resource=None, table_name=None):
    """
    Finds a model in DynamoDB by its 'model_name' using a scan.

    Accepts an optional `dynamodb_resource` to allow dependency injection
    (useful for tests or callers that want to control the boto3 resource).
    Note: A scan is inefficient on large tables. A GSI would be better.
    """
    try:
        dynamodb_resource = dynamodb_resource or dynamodb
        tbl = dynamodb_resource.Table(table_name or TABLE_NAME)
        # Some test fakes may not accept the same kwargs as boto3's Table.scan.
        # Try to call with a FilterExpression first (the real boto3 API), and
        # if the scan implementation doesn't accept that kwarg, fall back to
        # scanning without args and filtering in Python.
        try:
            response = tbl.scan(FilterExpression=Attr("model_name").eq(name))
            items = response.get("Items", [])
        except TypeError:
            response = tbl.scan()
            items = [
                it for it in response.get("Items", []) if it.get("model_name") == name
            ]
        if items:
            return items  # Assume model_name is unique
        return None
    except Exception as e:
        logger.error(f"Error scanning for model_name '{name}': {e}")
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


def attribute_is_not_none(model_id, attribute_name):
    """
    Checks if a given attribute for a model exists and is not None.

    :param model_id: The ID of the model to check.
    :param attribute_name: The name of the attribute to check.
    :return: True if the attribute exists and is not None, False otherwise.
    """
    item = get_model_by_id(model_id)
    if not item:
        logger.warning(f"Item with id '{model_id}' not found for attribute check.")
        return False

    # .get() returns None if the key doesn't exist, so this single
    # check handles both missing keys and keys with a value of None.
    is_present_and_not_none = item.get(attribute_name) is not None
    logger.info(
        f"Check for attribute '{attribute_name}' \
            on item '{model_id}': {is_present_and_not_none}"
    )
    return is_present_and_not_none


def get_attribute_value(model_id, attribute_name):
    """
    Gets the value of a specific attribute for a given model ID.

    :param model_id: The ID of the model to check.
    :param attribute_name: The name of the attribute to retrieve.
    :return: The value of the attribute, or None if the item or attribute doesn't exist.
    """
    item = get_model_by_id(model_id)
    if not item:
        logger.warning(f"Item with id '{model_id}' not found for attribute retrieval.")
        return None

    value = item.get(attribute_name)
    return value


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
