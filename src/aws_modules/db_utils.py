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
        return Decimal(str(obj))
    else:
        return obj


def save_model_metadata(name, version, s3_key, scores, artifact_type="model"):
    if not table:
        logger.error("DynamoDB table is not initialized. Check env var.")
        return None

    try:
        new_id = str(uuid.uuid4())
        scores_with_decimal = floats_to_decimals(scores)

        item = {
            "id": new_id,
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
    try:
        dynamodb_resource = dynamodb_resource or dynamodb
        tbl = dynamodb_resource.Table(table_name or TABLE_NAME)
        try:
            response = tbl.scan(FilterExpression=Attr("model_name").eq(name))
            items = response.get("Items", [])
        except TypeError:
            response = tbl.scan()
            items = [
                it for it in response.get("Items", []) if it.get("model_name") == name
            ]
        if items:
            return items
        return None
    except Exception as e:
        logger.error(f"Error scanning for model_name '{name}': {e}")
        return None


def get_model_by_repo_id(repo_id):
    try:
        tbl = dynamodb.Table(TABLE_NAME)
        response = tbl.scan(FilterExpression=Attr("repo_id").eq(repo_id))
        items = response.get("Items", [])
        if items:
            return items[0]
        return None
    except Exception as e:
        logger.error(f"Error scanning for repo_id '{repo_id}': {e}")
        return None


def get_all_artifacts_by_type(artifact_type):
    """
    Scans the DB for all items of a specific type (e.g. 'code', 'dataset').
    Returns a list of items.
    """
    if not table:
        logger.error("DynamoDB table is not initialized. Check env var.")
        return []
    
    try:
        response = table.scan(FilterExpression=Attr("type").eq(artifact_type))
        items = response.get("Items", [])
        
        while 'LastEvaluatedKey' in response:
            response = table.scan(
                FilterExpression=Attr("type").eq(artifact_type),
                ExclusiveStartKey=response['LastEvaluatedKey']
            )
            items.extend(response.get("Items", []))
            
        # Normalize: ensure 'url' or 'source_url' is available as 'url'
        for item in items:
            if 'source_url' in item and 'url' not in item:
                item['url'] = item['source_url']
                
        return items
    except Exception as e:
        logger.error(f"Error scanning for type '{artifact_type}': {e}")
        return []


def attribute_is_not_none(model_id, attribute_name):
    item = get_model_by_id(model_id)
    if not item:
        logger.warning(f"Item with id '{model_id}' not found for attribute check.")
        return False

    is_present_and_not_none = item.get(attribute_name) is not None
    return is_present_and_not_none


def get_attribute_value(model_id, attribute_name):
    item = get_model_by_id(model_id)
    if not item:
        logger.warning(f"Item with id '{model_id}' not found for attribute retrieval.")
        return None

    value = item.get(attribute_name)
    return value


def delete_model_metadata(model_id):
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