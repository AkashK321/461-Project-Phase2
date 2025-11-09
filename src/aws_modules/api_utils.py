import json

def make_response(status_code, body):
    """
    Formats an API Gateway proxy response.
    """
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",  # Allow all origins
            "Access-Control-Allow-Headers": "*", # Allow all headers
            "Access-Control-Allow-Methods": "*", # Allow all methods
        },
        "body": json.dumps(body),
    }