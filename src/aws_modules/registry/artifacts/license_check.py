from aws_modules.db_utils import get_model_by_id
from scorer.metrics.license import license_check_bool, LicenseCheckError
from aws_modules.api_utils import make_response
from aws_modules.registry.system import logger


def license_check(art_id, body):
    if not art_id:
        return make_response(400, {"error": "Missing artifact ID in path"})

    if not isinstance(body, dict) or not body.get("github_url"):
        return make_response(
            400,
            {"error": "License check request malformed or missing github_url"},
        )

    github_url = body.get("github_url")

    # confirm the model exists in our registry first.
    model_item = get_model_by_id(art_id)
    if not model_item or model_item.get("type") != "model":
        return make_response(404, {"error": "The artifact could not be found."})

    model_url = model_item.get("source_url")
    if not model_url:
        # If ingest ever produced a model without a source_url,
        # treat it like upstream failure.
        return make_response(
            502,
            {"error": "External license information could not be retrieved."},
        )

    try:
        compatible = license_check_bool(model_url=model_url, github_url=github_url)
        return make_response(200, compatible)
    except LicenseCheckError as e:
        return make_response(e.status_code, {"error": e.message})
    except Exception:
        logger.exception("Unhandled error during license check")
        return make_response(
            502,
            {"error": "External license information could not be retrieved."},
        )
