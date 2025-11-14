import logging
import os
import boto3
from boto3.dynamodb.conditions import Attr
from huggingface_hub import HfApi, ModelCard
from aws_modules.db_utils import get_model_by_id

# Set up the logger for this utility file
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Get DDB resource
dynamodb = boto3.resource("dynamodb")
TABLE_NAME = os.getenv("DYNAMODB_TABLE_NAME", "")

def get_model_lineage_type(model_repo_id, base_model):
    api = HfApi()
    logger.info(f"Determining lineage type for model: {model_repo_id}")
    try:
        model_info = api.model_info(model_repo_id)
        
        # Get metadata (tags) and file list
        card_data = model_info.cardData or {}  # The YAML metadata in the README
        files = [f.rfilename for f in model_info.siblings]
        tags = model_info.tags or []

        if card_data.get("library_name") == "peft":
            return "Adapter", "model_card"
        if "adapter_config.json" in files:
            return "Adapter", "config_json"

        if any(f.endswith((".gguf", ".awq")) for f in files):
            return "Quantized", "filename"
        if "gptq" in tags or "GPTQ" in model_repo_id:
            return "Quantized", "tags"
        if any("quant" in f.lower() for f in files):
            return "Quantized", "filename"

        # If it has a 'base_model' but isn't an adapter/quant, it's likely
        # a full fine-tune or a merge. The source is the model card.
        if base_model:
            return "Fine-Tune", "model_card"

        # If no parent is found, it's a base model.
        return "Base", "inferred"

    except Exception as e:
        logger.error(
            f"Failed to determine lineage type for {model_repo_id}: {e}",
            exc_info=True,
        )
        return "Unknown", "error"


def get_base_model_from_card(model_repo_id):
    """
    Fetches the model card from Hugging Face and parses its
    metadata to find the 'base_model' key.
    """
    try:
        logger.info(f"Fetching model card for {model_repo_id}...")
        card = ModelCard.load(model_repo_id)
        base_model = card.data.get("base_model")

        if not base_model:
            logger.warning(f"No 'base_model' key in metadata for {model_repo_id}.")
            return None, "Base", "inferred"

        if isinstance(base_model, list):
            base_model_id = base_model[0]
        else:
            base_model_id = str(base_model)

        logger.info(f"Found base model in card: {base_model_id}")
        
        lineage_type, source = get_model_lineage_type(model_repo_id, base_model_id)

        return base_model_id, lineage_type, source

    except Exception as e:
        logger.warning(
            f"Could not retrieve base model from card for {model_repo_id}: {e}"
        )
        return None, "Unknown", "error"


def get_model_by_repo_id(repo_id):
    """
    Finds a model in DynamoDB by its 'repo_id' using a scan.
    """
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


def get_models_by_base_repo_id(base_repo_id):
    """
    Finds all models that have a specific base_model_repo_id.
    This uses a scan, which is inefficient on large tables.
    A GSI on 'base_model_repo_id' would be a performance improvement.
    """
    try:
        tbl = dynamodb.Table(TABLE_NAME)
        response = tbl.scan(FilterExpression=Attr("base_model_repo_id").eq(base_repo_id))
        items = response.get("Items", [])
        return items
    except Exception as e:
        logger.error(f"Error scanning for base_repo_id '{base_repo_id}': {e}")
        return []


def get_lineage_items_from_id(start_art_id):
    """Helper to traverse and return all items in a model's lineage."""
    lineage_items = []
    current_item = get_model_by_id(start_art_id)
    logger.info(f"Starting lineage trace from ID: {current_item}")

    while current_item:
        lineage_items.append(current_item)
        parent_repo_id = current_item.get("base_model_repo_id")

        if not parent_repo_id:
            logger.info("Reached root model (no parent repo ID).")
            break

        logger.info(f"Searching for parent: {parent_repo_id}")
        current_item = get_model_by_repo_id(parent_repo_id)

        if not current_item:
            logger.info(
                f"Parent '{parent_repo_id}' not found in registry. Ending trace."
            )

    return lineage_items


def get_descendant_items(start_repo_id):
    """
    Helper to recursively find all descendants of a model.
    This performs a breadth-first search.
    """
    if not start_repo_id:
        return []

    all_descendants = []
    queue = [start_repo_id]
    visited_repos = {start_repo_id}

    while queue:
        current_repo_id = queue.pop(0)
        logger.info(f"Finding children for: {current_repo_id}")

        # Find all direct children of the current model
        children = get_models_by_base_repo_id(current_repo_id)
        for child in children:
            child_repo_id = child.get("repo_id")
            if child_repo_id and child_repo_id not in visited_repos:
                all_descendants.append(child)
                queue.append(child_repo_id)
                visited_repos.add(child_repo_id)

    return all_descendants

def _calculate_treescore(base_model_repo_id):
    """
    Calculates the Treescore for a new model based on its parent's lineage.
    """
    if not base_model_repo_id:
        logger.info("No base model. Treescore is 0.")
        return 0.0

    parent_item = get_model_by_repo_id(base_model_repo_id)
    logger.info(f"Calculating treescore for base model repo ID: {base_model_repo_id}")
    if not parent_item:
        logger.info(f"Parent '{base_model_repo_id}' not in registry. Treescore is 0.")
        return 0.0

    parent_lineage = get_lineage_items_from_id(parent_item.get("id"))
    logger.info(f"Parent lineage: {parent_lineage}")

    if not parent_lineage:
        return 0.0

    total_score = 0.0
    num_parents = len(parent_lineage)

    for parent in parent_lineage:
        parent_scores = parent.get("scores", {})
        net_score = parent_scores.get("net_score", 0.0)
        total_score += float(net_score)

    logger.info(f"Treescore calculation: {total_score} / {num_parents}")
    return total_score / num_parents
