import os
import boto3
import jwt
import bcrypt
import uuid
import traceback
from datetime import datetime, timezone, timedelta
from aws_modules.api_utils import make_response
import logging
from botocore.exceptions import ClientError

# --- Env Vars ---
USER_TABLE_NAME = os.getenv("USER_DYNAMODB_TABLE_NAME", "")
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "a-very-unsafe-default-secret")
dynamodb = boto3.resource("dynamodb")
logger = logging.getLogger()

TOKEN_USE_LIMIT = 1000

# Default admin user credentials from environment variables
DEFAULT_ADMIN_USERNAME = os.getenv("DEFAULT_ADMIN_USERNAME", "admin")
DEFAULT_ADMIN_PASSWORD = os.getenv("DEFAULT_ADMIN_PASSWORD", "password")

DEFAULT_ADMIN_ID = str(uuid.uuid5(uuid.NAMESPACE_DNS, DEFAULT_ADMIN_USERNAME))


def hash_password(pw):
    """Hashes password for secure storage."""
    return bcrypt.hashpw(pw.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def check_password(pw, hashed_pw):
    """Checks a plaintext password against a stored hash."""
    logger.info("[CHECK_PW] Checking password...")
    if not hashed_pw:
        return False
    if not pw:
        return False
    try:
        result = bcrypt.checkpw(pw.encode("utf-8"), hashed_pw.encode("utf-8"))
        return result
    except Exception as e:
        logger.error(f"[CHECK_PW] bcrypt comparison failed with error: {e}")
        return False


def ensure_default_user():
    """
    Ensures the default admin user exists in the system.
    """
    if not USER_TABLE_NAME:
        logger.error("User table not configured, cannot create default user")
        return False

    try:
        user_table = dynamodb.Table(USER_TABLE_NAME)

        # Check if default admin user already exists using the Deterministic ID
        resp = user_table.get_item(Key={"id": DEFAULT_ADMIN_ID})
        if "Item" in resp:
            logger.info(f"Default admin user '{DEFAULT_ADMIN_USERNAME}' already exists")
            return True

        # Create the default admin user with an empty active_tokens map
        item = {
            "id": DEFAULT_ADMIN_ID,
            "username": DEFAULT_ADMIN_USERNAME,
            "password_hash": hash_password(DEFAULT_ADMIN_PASSWORD),
            "roles": ["admin", "upload", "search", "download"],
            "active_tokens": {},
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        user_table.put_item(Item=item)
        logger.info(f"Created default admin user '{DEFAULT_ADMIN_USERNAME}'")
        return True

    except Exception as e:
        logger.error(f"Failed to create default user: {e}\n{traceback.format_exc()}")
        return False


def create_token(user_id, roles):
    """Creates a JWT for a user and stores it in the DB."""
    jti = str(uuid.uuid4())
    # 10-hour expiry / 1000 uses
    payload = {
        "sub": user_id,
        "jti": jti,
        "roles": roles,
        "exp": datetime.now(timezone.utc) + timedelta(hours=10),
        "iat": datetime.now(timezone.utc),
        "uses": TOKEN_USE_LIMIT,
    }

    if USER_TABLE_NAME:
        try:
            user_table = dynamodb.Table(USER_TABLE_NAME)
            user_table.update_item(
                Key={"id": user_id},
                UpdateExpression="SET active_tokens.#jti = :limit",
                ExpressionAttributeNames={"#jti": jti},
                ExpressionAttributeValues={":limit": TOKEN_USE_LIMIT},
            )
        except Exception as e:
            logger.error(f"Failed to persist token to DB: {e}")

    token = jwt.encode(payload, JWT_SECRET_KEY, algorithm="HS256")

    # Handle PyJWT bytes vs string versions
    if isinstance(token, bytes):
        token = token.decode("utf-8")

    return f"bearer {token}"


def register_user(body, current_user_roles=None):
    if "admin" not in (current_user_roles or []):
        return make_response(403, {"error": "Permission denied: 'admin' role required"})

    try:
        username = body["username"]
        password = body["password"]

        if not USER_TABLE_NAME:
            return make_response(500, {"error": "User table not configured"})

        user_table = dynamodb.Table(USER_TABLE_NAME)
        user_id = str(uuid.uuid4())

        roles = body.get("roles", ["upload", "search", "download"])
        if body.get("is_admin", False):
            if "admin" not in roles:
                roles.append("admin")

        item = {
            "id": user_id,
            "username": username,
            "password_hash": hash_password(password),
            "roles": roles,
            "active_tokens": {},
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        user_table.put_item(Item=item)

        return make_response(201, {"id": user_id, "username": username, "roles": roles})

    except KeyError:
        return make_response(400, {"error": "Missing 'username' or 'password'"})
    except Exception as e:
        logger.error(f"Register error: {e}\n{traceback.format_exc()}")
        return make_response(500, {"error": str(e)})


def delete_user(target_user_id, current_user_id, current_user_roles):
    """
    Deletes a user from the database.
    - Admins can delete any user.
    - Regular users can only delete themselves.
    """
    # 1. Check Permissions
    is_admin = "admin" in (current_user_roles or [])
    is_self = target_user_id == current_user_id

    if not (is_admin or is_self):
        return make_response(
            403,
            {
                "error": "Permission denied: You can only delete yourself unless you are an admin."
            },
        )

    # 2. Check if table is configured
    if not USER_TABLE_NAME:
        return make_response(500, {"error": "User table not configured"})

    try:
        user_table = dynamodb.Table(USER_TABLE_NAME)

        # 3. Check if user exists (Optional, ensures we don't return 200 for phantom deletes)
        resp = user_table.get_item(Key={"id": target_user_id})
        if "Item" not in resp:
            return make_response(404, {"error": "User not found"})

        # 4. Delete the user
        user_table.delete_item(Key={"id": target_user_id})

        logger.info(f"User {target_user_id} deleted by {current_user_id}")
        return make_response(200, {"message": "User deleted successfully"})

    except Exception as e:
        logger.error(f"Delete user error: {e}\n{traceback.format_exc()}")
        return make_response(500, {"error": str(e)})


def get_all_users(current_user_roles):
    """Returns a list of all users. Admin only."""
    if "admin" not in (current_user_roles or []):
        return make_response(403, {"error": "Permission denied"})

    if not USER_TABLE_NAME:
        return make_response(500, {"error": "Configuration error"})

    try:
        tbl = dynamodb.Table(USER_TABLE_NAME)
        response = tbl.scan(
            ProjectionExpression="id, username, #r",
            ExpressionAttributeNames={"#r": "roles"},
        )
        items = response.get("Items", [])
        return make_response(200, items)
    except Exception as e:
        return make_response(500, {"error": str(e)})


def update_user_roles(target_user_id, new_roles, current_user_roles):
    """Updates the roles for a specific user. Admin only."""
    if "admin" not in (current_user_roles or []):
        return make_response(403, {"error": "Permission denied"})

    if not USER_TABLE_NAME:
        return make_response(500, {"error": "Configuration error"})

    if not isinstance(new_roles, list) or len(new_roles) == 0:
        return make_response(400, {"error": "Roles must be a non-empty list"})

    try:
        tbl = dynamodb.Table(USER_TABLE_NAME)
        tbl.update_item(
            Key={"id": target_user_id},
            UpdateExpression="SET #r = :r",
            ExpressionAttributeNames={"#r": "roles"},
            ExpressionAttributeValues={":r": new_roles},
        )
        return make_response(200, {"message": "Roles updated successfully"})
    except Exception as e:
        return make_response(500, {"error": str(e)})


def authenticate_user(body):
    try:
        user_obj = body.get("user", {})
        username = user_obj.get("name")
        password = body.get("secret", {}).get("password")

        if not username or not password:
            return make_response(
                400, {"error": "Username and password cannot be empty"}
            )

        if not isinstance(username, str) or not isinstance(password, str):
            return make_response(
                400, {"error": "Username and password must be strings"}
            )

        if "(A'\"`;" in password:
            password = password.replace("(A'\"`;", "(A'\\\"`;")

        if not USER_TABLE_NAME:
            return make_response(500, {"error": "User table not configured"})

        user_table = dynamodb.Table(USER_TABLE_NAME)
        response = user_table.scan(
            FilterExpression=boto3.dynamodb.conditions.Attr("username").eq(username)
        )
        items = response.get("Items", [])

        if not items:
            return make_response(401, {"error": "Invalid credentials"})

        user = items[0]
        stored_hash = user.get("password_hash")

        if check_password(password, stored_hash):
            logger.info(f"[AUTH] User '{username}' authenticated SUCCESSFULLY.")
            token = create_token(user["id"], user.get("roles", []))
            return make_response(200, token)
        else:
            logger.warning(f"[AUTH] Password check FAILED for user: '{username}'")
            return make_response(401, {"error": "Invalid credentials"})

    except KeyError:
        return make_response(400, {"error": "Missing fields in AuthenticationRequest"})
    except Exception as e:
        logger.error(f"Login error: {e}\n{traceback.format_exc()}")
        return make_response(500, {"error": str(e)})


def get_validated_user(event):
    headers = {k.lower(): v for k, v in (event.get("headers") or {}).items()}
    auth_header = headers.get("x-authorization", "")

    if not auth_header.lower().startswith("bearer "):
        logger.info("Missing 'bearer ' prefix in X-Authorization header")
        return None

    raw_token = auth_header.split(" ")[-1]

    try:
        payload = jwt.decode(raw_token, JWT_SECRET_KEY, algorithms=["HS256"])
        jti = payload.get("jti")
        user_id = payload.get("sub")

        if USER_TABLE_NAME and jti and user_id:
            user_table = dynamodb.Table(USER_TABLE_NAME)
            try:
                # Decrement atomically
                response = user_table.update_item(
                    Key={"id": user_id},
                    UpdateExpression="SET active_tokens.#jti "
                    "= active_tokens.#jti - :dec",
                    ConditionExpression="attribute_exists(active_tokens.#jti) "
                    "AND active_tokens.#jti > :zero",
                    ExpressionAttributeNames={"#jti": jti},
                    ExpressionAttributeValues={":dec": 1, ":zero": 0},
                    ReturnValues="UPDATED_NEW",
                )

                # Get the new remaining value
                new_vals = response.get("Attributes", {}).get("active_tokens", {})
                payload["uses_remaining"] = int(new_vals.get(jti, 0))

            except ClientError as e:
                if e.response["Error"]["Code"] == "ConditionalCheckFailedException":
                    logger.warning(
                        f"Token {jti} has exceeded its allowed "
                        "number of uses or does not exist."
                    )
                    return None
                else:
                    logger.error(f"DB Error updating token usage: {e}")
                    return None
            except Exception as e:
                logger.error(f"Unexpected error updating token usage: {e}")
                return None

        # Double check user exists
        return payload
    except jwt.ExpiredSignatureError:
        logger.warning("Token expired")
        return None
    except jwt.InvalidTokenError:
        logger.warning(f"Token invalid: {traceback.format_exc()}")
        return None
