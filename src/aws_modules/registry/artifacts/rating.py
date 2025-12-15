"""
Artifact rating utilities.
"""

import json

from aws_modules.db_utils import get_model_by_id
from aws_modules.api_utils import make_response

from aws_modules.registry.system import logger
from aws_modules.registry.context import lambda_client, SCORER_FUNCTION_NAME


def rate_model(art_id):
    """
    Handle GET /artifact/model/{id}/rate
    Retrieves and returns the stored scores for a given model artifact.
    Falls back to invoking scorer if DB scores are missing.

    :param art_id: The ID of the artifact to rate.
    :return: A response with the rating scores.
    """
    item = get_model_by_id(art_id)
    if not item:
        return make_response(404, {"error": "Artifact does not exist."})

    # 1. Try DB scores first
    scores = item.get("scores", {})

    # 2. Fallback: Invoke scorer if scores missing
    if not scores:
        scorer_function_name = SCORER_FUNCTION_NAME
        if not scorer_function_name:
            logger.error("Scorer function not configured, cannot validate metrics")
            return make_response(500, {"error": "Metric scoring service not available"})

        logger.info(f"Invoking scorer function: {scorer_function_name}")
        try:
            url = item.get("source_url")
            if not url:
                logger.error(f"No source URL for artifact {art_id}")
                # Cannot rate without URL
                return make_response(500, {"error": "Failed to calculate metrics"})

            logger.info(f"Calculating scores for artifact {art_id} at URL {url}")
            scorer_payload = json.dumps({"urls": [url]})

            response = lambda_client.invoke(
                FunctionName=scorer_function_name,
                InvocationType="RequestResponse",
                Payload=scorer_payload,
            )
            response_payload = json.loads(response["Payload"].read().decode())
            if response_payload.get("statusCode") == 200:
                scores_list = json.loads(response_payload["body"])
                if scores_list:
                    scores = scores_list[0]
            else:
                logger.error(f"Scorer function returned error: {response_payload}")
                return make_response(500, {"error": "Failed to calculate metrics"})
        except Exception as e:
            logger.error(f"Failed to invoke or parse scorer response: {e}")
            return make_response(500, {"error": "Failed to calculate metrics"})

    if not scores:
        logger.error(f"Scores not found for artifact {art_id}, but item exists.")
        return make_response(
            500,
            {
                "error": "The artifact rating system encountered "
                "an error while computing at least one metric."
            },
        )
    else:
        for metric, score in scores.items():
            logger.info(f"Metric: {metric}, Score: {score}")
            if score is None:
                logger.warning(f"Score for metric '{metric}' is None.")
                return make_response(
                    500,
                    {
                        "error": "The artifact rating system encountered an error "
                        "while computing at least one metric."
                    },
                )

    return make_response(200, scores)
