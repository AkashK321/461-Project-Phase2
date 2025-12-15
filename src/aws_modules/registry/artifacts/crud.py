"""
CRUD operations for artifacts.
"""

from aws_modules.api_utils import make_response
from aws_modules.db_utils import get_model_by_id, delete_model_metadata
from aws_modules.s3_utils import generate_presigned_download_url, delete_model
from aws_modules.registry.system import logger


def get_artifact(art_id):
    """Retrieve metadata and download URL for a given artifact ID.

    :param art_id: The ID of the artifact to retrieve.
    :return: A response dictionary with metadata and data.
    """
    item = get_model_by_id(art_id)
    if not item:
        return make_response(404, {"error": "Artifact does not exist."})

    metadata = {
        "name": item.get("model_name"),
        "id": item.get("id"),
        "type": item.get("type"),
    }
    s3_key = item.get("s3_key")
    download_url = None
    if s3_key:
        download_url = generate_presigned_download_url(s3_key)
    else:
        logger.warning(f"Artifact {art_id} has no 's3_key' to generate download URL")

    data = {"url": item.get("source_url")}

    if download_url:
        data["download_url"] = download_url

    return make_response(200, {"metadata": metadata, "data": data})


def delete_artifact(artifact_type, art_id, user_roles):
    """Delete an artifact from the registry and S3.

    :param artifact_type: The type of the artifact ('model', 'dataset', 'code').
    :param art_id: The ID of the artifact to delete.
    :param user_roles: The roles of the user performing the deletion.
    :return: A response dictionary indicating success or failure.
    """
    if artifact_type not in {"model", "dataset", "code"} or not art_id:
        return make_response(
            400,
            {"error": "missing field in artifact_type, artifact_id, " "or invalid"},
        )

    # Permission: only admins can delete artifacts
    if "admin" not in user_roles:
        return make_response(403, {"error": "Permission denied"})

    item = get_model_by_id(art_id)
    if not item or item.get("type") != artifact_type:
        return make_response(404, {"error": "Artifact does not exist."})

    s3_key = item.get("s3_key")
    if s3_key:
        if not delete_model(s3_key):
            return make_response(500, {"error": "Failed to delete artifact content."})

    if not delete_model_metadata(art_id):
        return make_response(500, {"error": "Failed to delete artifact metadata."})

    return make_response(200, {"message": "Artifact is deleted."})
