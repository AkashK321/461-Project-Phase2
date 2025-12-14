import os
import uuid
import logging

import boto3

from aws_modules.auth import (
    authenticate_user,
    get_validated_user,
    register_user,
    delete_user,
    get_all_users,
    update_user_roles,
    ensure_default_user,
)

# logging setup
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

# shared AWS clients/resources
dynamodb = boto3.resource("dynamodb")
s3 = boto3.client("s3")

TABLE_NAME = os.getenv("DYNAMODB_TABLE_NAME", "")
BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "")
USER_TABLE_NAME = os.getenv("USER_DYNAMODB_TABLE_NAME", "")
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "a-very-unsafe-default-secret")

# Global flag to track if initialization has been performed
_initialized = False


def initialize_system():
    global _initialized
    if _initialized:
        return
    if USER_TABLE_NAME and JWT_SECRET_KEY:
        ensure_default_user()
    _initialized = True


def reset_state(restore_jti=None):
    # 1. Wipe Registry
    tbl = dynamodb.Table(TABLE_NAME)
    scan = tbl.scan(ProjectionExpression="#i", ExpressionAttributeNames={"#i": "id"})
    ids = [it["id"] for it in scan.get("Items", [])]
    if ids:
        with tbl.batch_writer() as batch:
            for _id in ids:
                batch.delete_item(Key={"id": _id})

    # 2. Wipe S3
    if BUCKET_NAME:
        prefixes = ["models/", "artifacts/"]
        paginator = s3.get_paginator("list_objects_v2")
        for prefix in prefixes:
            for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix=prefix):
                objs = [{"Key": o["Key"]} for o in page.get("Contents", [])]
                if objs:
                    s3.delete_objects(Bucket=BUCKET_NAME, Delete={"Objects": objs})

    # 3. Wipe User Table and Restore
    if USER_TABLE_NAME:
        user_tbl = dynamodb.Table(USER_TABLE_NAME)
        scan = user_tbl.scan(
            ProjectionExpression="#i", ExpressionAttributeNames={"#i": "id"}
        )
        user_ids = [it["id"] for it in scan.get("Items", [])]
        if user_ids:
            with user_tbl.batch_writer() as batch:
                for _id in user_ids:
                    batch.delete_item(Key={"id": _id})

        ensure_default_user()

        if restore_jti:
            default_admin_user = os.getenv("DEFAULT_ADMIN_USERNAME", "admin")
            default_admin_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, default_admin_user))
            try:
                user_tbl.update_item(
                    Key={"id": default_admin_id},
                    UpdateExpression="SET active_tokens.#j = :l",
                    ExpressionAttributeNames={"#j": restore_jti},
                    ExpressionAttributeValues={":l": 1000},
                )
            except Exception as e:
                logger.error(f"Failed to restore JTI after reset: {e}")

    return {"reset": "ok"}
