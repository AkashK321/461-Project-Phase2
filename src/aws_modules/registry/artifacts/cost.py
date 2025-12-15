"""
Artifact cost calculation utilities.
"""

from aws_modules.api_utils import make_response
from aws_modules.db_utils import get_model_by_id
from utils.lineage_utils import get_lineage_items_from_id
import logging
import json


# logging setup
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

from aws_modules.s3_utils import (
    get_object_size,
)


def calculate_artifact_cost(art_id, query_params):
    """
    Handle GET /artifact/{type}/{id}/cost
    Returns the size cost of the artifact in MB.
    If 'dependency=true' (query param),
    it includes dependencies (ancestors) in the cost.

    :param art_id: The ID of the artifact.
    :param query_params: Query parameters dictionary.
    :return: A response with cost data.
    """
    logger.info(
        f"Entering calculate_artifact_cost \
        with art_id='{art_id}', params={query_params}"
    )

    # Parse query params
    dependency_param = query_params.get("dependency", "false").lower() == "true"

    # Get the target artifact
    target_item = get_model_by_id(art_id)
    if not target_item:
        logger.warning(f"Artifact {art_id} not found in database.")
        response = make_response(404, {"error": "Artifact does not exist."})
        logger.info(f"calculate_artifact_cost response for {art_id}: {response}")
        return response

    logger.info(f"Artifact found: {target_item.get('model_name')}")

    # Gather all involved artifacts
    all_items = {art_id: target_item}

    if dependency_param:
        ancestors = get_lineage_items_from_id(art_id)
        if ancestors:
            for anc in ancestors:
                if anc.get("id"):
                    all_items[anc["id"]] = anc

    logger.info(
        f"Cost calculation includes {len(all_items)} \
        artifacts (dependency={dependency_param})"
    )

    # Helper to calculate size in MB
    def get_size_mb(item_id, item_data):
        s3_key = item_data.get("s3_key")
        if not s3_key:
            logger.warning(f"S3 key missing for artifact {item_id}")
            return 0.0

        size_bytes = get_object_size(s3_key)
        if size_bytes is None:
            logger.warning(f"Failed to retrieve size for S3 key: {s3_key}")
            return 0.0

        return float(size_bytes) / (1024 * 1024)

    # Calculate sizes for all items
    item_sizes_mb = {}
    for i_id, i_data in all_items.items():
        item_sizes_mb[i_id] = get_size_mb(i_id, i_data)
        logger.info(f"Size for {i_id}: {item_sizes_mb[i_id]} MB")

    response_data = {}

    if not dependency_param:
        # Case 1: No dependencies requested
        size = item_sizes_mb[art_id]
        response_data[art_id] = {"total_cost": size}
    else:
        # Case 2: Dependencies requested
        total_sum = sum(item_sizes_mb.values())

        # 1. The root/target artifact
        response_data[art_id] = {
            "standalone_cost": item_sizes_mb[art_id],
            "total_cost": total_sum,
        }

        # 2. The dependencies
        for i_id, size in item_sizes_mb.items():
            if i_id == art_id:
                continue
            response_data[i_id] = {"standalone_cost": size, "total_cost": size}

    response = make_response(200, response_data)
    logger.info(f"calculate_artifact_cost final response: {json.dumps(response_data)}")
    return response
