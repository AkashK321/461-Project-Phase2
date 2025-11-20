import json
import decimal


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
    }

    final_headers.update(headers)

    return {
        "statusCode": status,
        "headers": final_headers,
        "body": json.dumps(body, cls=DecimalEncoder),
    }
