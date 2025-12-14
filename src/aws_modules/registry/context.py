import os
import boto3
from botocore.config import Config

os.environ["HF_HOME"] = "/tmp/huggingface"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/tmp/huggingface/hub"
os.environ["HF_ASSETS_CACHE"] = "/tmp/huggingface/assets"

TABLE_NAME = os.getenv("DYNAMODB_TABLE_NAME", "")
BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "")
USER_TABLE_NAME = os.getenv("USER_DYNAMODB_TABLE_NAME", "")
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "a-very-unsafe-default-secret")
SCORER_FUNCTION_NAME = os.getenv("SCORER_FUNCTION_NAME", "scorer_function")

DEFAULT_PAGE_SIZE = int(os.getenv("DEFAULT_PAGE_SIZE", "10"))

FEATURE_FLAG_FORCE_INGESTION = (
    os.getenv("FEATURE_FLAG_FORCE_INGESTION", "false").lower() == "true"
)

# shared AWS clients/resources
dynamodb = boto3.resource("dynamodb")
s3 = boto3.client("s3")

# Configure Lambda client with a timeout
lambda_config = Config(
    read_timeout=480, connect_timeout=60, retries={"max_attempts": 1}
)
lambda_client = boto3.client("lambda", config=lambda_config)
