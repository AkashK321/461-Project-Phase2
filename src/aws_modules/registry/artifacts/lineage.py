"""
Lineage graph utilities for artifacts.
"""

from aws_modules.db_utils import get_model_by_id
from utils.lineage_utils import get_lineage_items_from_id, get_descendant_items
from aws_modules.api_utils import make_response
from aws_modules.registry.system import logger


def get_lineage_graph(start_art_id):
    """
    Handle GET /artifact/model/{id}/lineage
    Constructs and returns the lineage graph for a given model ID.

    :param start_art_id: The ID of the starting artifact.
    :return: A response with nodes and edges of the lineage graph.
    """
    start_item = get_model_by_id(start_art_id)
    if not start_item:
        return make_response(404, {"error": "Artifact does not exist."})

    # --- Build the full graph: ancestors + start_node + descendants ---
    ancestors = get_lineage_items_from_id(start_art_id)
    start_repo_id = start_item.get("repo_id")

    # Per spec, if metadata is malformed for lineage (e.g., missing repo_id),
    # return a 400 error.
    if not start_repo_id:
        return make_response(
            400,
            {
                "error": "The lineage graph cannot be computed \
                because the artifact metadata is missing or malformed."
            },
        )
    descendants = get_descendant_items(start_repo_id)

    # Combine all items, ensuring no duplicates
    all_items = {item["id"]: item for item in ancestors}
    all_items[start_item["id"]] = start_item
    for item in descendants:
        all_items[item["id"]] = item

    # --- Construct nodes and edges from all items ---
    nodes = []
    edges = []
    for item_id, item in all_items.items():
        nodes.append(
            {
                "artifact_id": item_id,
                "name": item.get("model_name"),
                "source": item.get("lineage_source", "inferred"),
            }
        )

        # If the item has a parent, create an edge
        parent_repo_id = item.get("base_model_repo_id")
        if parent_repo_id:
            # Find the parent item in our collected items
            parent_item = None
            for p_item in all_items.values():
                if p_item.get("repo_id") == parent_repo_id:
                    parent_item = p_item
                    break

            if parent_item:
                edges.append(
                    {
                        "from_node_artifact_id": parent_item.get("id"),
                        "to_node_artifact_id": item_id,
                        "relationship": item.get("lineage_type", "fine_tuned_from"),
                    }
                )
            else:
                # This case can happen if a parent exists but is not in the registry
                logger.warning(
                    f"Parent model with repo_id '{parent_repo_id}' not \
                        found in registry for child '{item.get('repo_id')}'"
                )

    return make_response(200, {"nodes": nodes, "edges": edges})
