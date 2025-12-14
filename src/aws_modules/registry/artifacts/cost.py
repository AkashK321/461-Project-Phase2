from aws_modules.api_utils import make_response
from aws_modules.db_utils import get_model_by_id
from utils.lineage_utils import get_lineage_items_from_id
from aws_modules.registry.system import logger


def calculate_artifact_cost(art_id, query_params):
    """
    Handle GET /artifact/{type}/{id}/cost
    Returns the size cost of the artifact in MB.
    If 'dependency=true' (query param),
    it includes dependencies (ancestors) in the cost.
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
