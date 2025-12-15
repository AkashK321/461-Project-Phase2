"""
Artifact retrieval by name utilities.
"""

from aws_modules.db_utils import get_model_by_model_name
from aws_modules.api_utils import make_response
from aws_modules.registry.system import dynamodb, TABLE_NAME


def get_artifacts_by_name(name):
    """
    Handle GET /artifact/byName/{name}
    Finds and returns metadata for all artifacts matching a given name.

    :param name: The name of the artifacts to search for.
    :return: A response with a list of artifact metadata.
    """
    # Pass this module's dynamodb to allow callers/tests to inject a fake resource.
    items = get_model_by_model_name(
        name, dynamodb_resource=dynamodb, table_name=TABLE_NAME
    )

    if not items:
        return make_response(404, {"error": "No such artifact."})

    # Format the items into the ArtifactMetadata schema
    metadata_list = [
        {
            "name": item.get("model_name"),
            "id": item.get("id"),
            "type": item.get("type"),
        }
        for item in items
    ]

    return make_response(200, metadata_list)
