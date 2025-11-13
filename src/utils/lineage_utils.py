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

        is_adapter = (
            card_data.get('library_name') == 'peft' or 
            'adapter_config.json' in files
        )

        if is_adapter:
            return "Adapter"
        
        is_gguf = any(f.endswith('.gguf') for f in files)
        is_awq = any(f.endswith('.awq') for f in files)
        is_gptq = 'gptq' in tags or 'GPTQ' in model_repo_id
        has_quant_filename = any("quant" in f.lower() for f in files)

        if is_gguf or is_awq or is_gptq or has_quant_filename:
            return "Quantized"

        # --- CHECK 3: IS IT A FINE-TUNE? ---
        # If it has a 'base_model' tag but isn't an adapter/quant, it's likely a full fine-tune
        # or a merge.
        if base_model:
            return "Fine-Tune"

        return "Base"

    except Exception as e:
        return "Unknown"


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
            return None

        if isinstance(base_model, list):
            base_model_id = base_model[0]
        else:
            base_model_id = str(base_model)

        logger.info(f"Found base model in card: {base_model_id}")
        
        lineage_type = get_model_lineage_type(model_repo_id, base_model_id)

        return base_model_id, lineage_type

    except Exception as e:
        logger.warning(
            f"Could not retrieve base model from card for {model_repo_id}: {e}"
        )
        return None, None


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


def get_lineage_items_from_id(start_art_id):
    """Helper to traverse and return all items in a model's lineage."""
    lineage_items = []
    current_item = get_model_by_id(start_art_id)

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


def _calculate_treescore(base_model_repo_id):
    """
    Calculates the Treescore for a new model based on its parent's lineage.
    """
    if not base_model_repo_id:
        logger.info("No base model. Treescore is 0.")
        return 0.0

    parent_item = get_model_by_repo_id(base_model_repo_id)
    if not parent_item:
        logger.info(f"Parent '{base_model_repo_id}' not in registry. Treescore is 0.")
        return 0.0

    parent_lineage = get_lineage_items_from_id(parent_item.get("id"))

    if not parent_lineage:
        return 0.0

    total_score = 0.0
    num_parents = len(parent_lineage)

    for parent in parent_lineage:
        parent_scores = parent.get("scores", {})
        net_score = parent_scores.get("net_score", 0.0)
        total_score += net_score

    logger.info(f"Treescore calculation: {total_score} / {num_parents}")
    return total_score / num_parents
