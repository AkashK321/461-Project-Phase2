"""Small API utility helpers.

Provides JSON response shaping and a Decimal encoder for DynamoDB items so
responses can be serialized to JSON for API Gateway.
"""

import json
import decimal
import logging


class DecimalEncoder(json.JSONEncoder):
    """
    Helper class to convert a DynamoDB item's Decimal types to JSON.
    """

    def default(self, o):
        if isinstance(o, decimal.Decimal):
            # Convert Decimal to float for JSON serialization
            return float(o)
        return super(DecimalEncoder, self).default(o)


def make_response(status, body, headers=None):
    """
    Wraps a status code, body, and headers into a valid API Gateway response.
    """
    if headers is None:
        headers = {}

    # Standard headers for CORS and content type
    final_headers = {
        "Content-Type": "application/json",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "*",
        "Access-Control-Allow-Headers": "*",
        "Access-Control-Expose-Headers": "Offset",
    }

    final_headers.update(headers)

    # Log the outgoing response headers and status for debugging
    try:
        logging.getLogger(__name__).info(
            "make_response returning status=%s headers=%s",
            status,
            final_headers,
        )
    except Exception:
        # Don't let logging failures break the response
        pass

    return {
        "statusCode": status,
        "headers": final_headers,
        "body": json.dumps(body, cls=DecimalEncoder),
    }
