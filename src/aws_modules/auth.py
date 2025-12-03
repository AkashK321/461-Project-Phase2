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

# Default credentials
DEFAULT_ADMIN_USERNAME = os.getenv("DEFAULT_ADMIN_USERNAME", "admin")
DEFAULT_ADMIN_PASSWORD = os.getenv("DEFAULT_ADMIN_PASSWORD", "password")

# Deterministic Admin ID (UUID5) so the ID stays the same across resets
DEFAULT_ADMIN_ID = str(uuid.uuid5(uuid.NAMESPACE_DNS, DEFAULT_ADMIN_USERNAME))


def hash_password(pw):
    """Hashes password for secure storage."""
    return bcrypt.hashpw(pw.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def check_password(pw, hashed_pw):
    """Checks a plaintext password against a stored hash."""
    logger.info("[CHECK_PW] Checking password...")
    if not hashed_pw or not pw:
        return False
    try:
        return bcrypt.checkpw(pw.encode("utf-8"), hashed_pw.encode("utf-8"))
    except Exception as e:
        logger.error(f"[CHECK_PW] bcrypt error: {e}")
        return False


def ensure_default_user():
    """Ensures the default admin user exists."""
    if not USER_TABLE_NAME:
        logger.error("User table not configured")
        return False

    try:
        user_table = dynamodb.Table(USER_TABLE_NAME)
        
        resp = user_table.get_item(Key={"id": DEFAULT_ADMIN_ID})
        if "Item" in resp:
            logger.info(f"Default admin '{DEFAULT_ADMIN_USERNAME}' exists")
            return True

        # Create default admin with empty active_tokens map
        item = {
            "id": DEFAULT_ADMIN_ID,
            "username": DEFAULT_ADMIN_USERNAME,
            "password_hash": hash_password(DEFAULT_ADMIN_PASSWORD),
            "roles": ["admin", "upload", "search", "download"],
            "active_tokens": {},  
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        user_table.put_item(Item=item)
        logger.info(f"Created default admin '{DEFAULT_ADMIN_USERNAME}'")
        return True

    except Exception as e:
        logger.error(f"Failed to create default user: {e}")
        return False


def create_token(user_id, roles):
    """Creates a JWT and adds it to the user's active_tokens map."""
    jti = str(uuid.uuid4())
    exp_time = datetime.now(timezone.utc) + timedelta(hours=10)
    
    payload = {
        "sub": user_id,
        "jti": jti,
        "roles": roles,
        "exp": exp_time,
        "iat": datetime.now(timezone.utc),
        "uses": TOKEN_USE_LIMIT,
    }

    if USER_TABLE_NAME:
        try:
            user_table = dynamodb.Table(USER_TABLE_NAME)
            # Add the new token JTI to the active_tokens map with 1000 uses
            user_table.update_item(
                Key={"id": user_id},
                UpdateExpression="SET active_tokens.#jti = :limit",
                ExpressionAttributeNames={"#jti": jti},
                ExpressionAttributeValues={":limit": TOKEN_USE_LIMIT}
            )
        except Exception as e:
            logger.error(f"Failed to persist token to DB: {e}")

    token = jwt.encode(payload, JWT_SECRET_KEY, algorithm="HS256")
    
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


def authenticate_user(body):
    try:
        user_obj = body.get("user", {})
        username = user_obj.get("name")
        password = body.get("secret", {}).get("password")

        if not username or not password:
            return make_response(400, {"error": "Missing credentials"})

        if not isinstance(username, str) or not isinstance(password, str):
            return make_response(400, {"error": "Credentials must be strings"})

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
        if check_password(password, user.get("password_hash")):
            logger.info(f"[AUTH] User '{username}' authenticated.")
            token = create_token(user["id"], user.get("roles", []))
            return make_response(200, token)
        else:
            return make_response(401, {"error": "Invalid credentials"})

    except KeyError:
        return make_response(400, {"error": "Bad Request"})
    except Exception as e:
        logger.error(f"Login error: {e}")
        return make_response(500, {"error": str(e)})


def get_validated_user(event):
    headers = {k.lower(): v for k, v in (event.get("headers") or {}).items()}
    auth_header = headers.get("x-authorization", "")

    if not auth_header.lower().startswith("bearer "):
        return None

    raw_token = auth_header.split(" ", 1)[-1]

    try:
        payload = jwt.decode(raw_token, JWT_SECRET_KEY, algorithms=["HS256"])
        jti = payload.get("jti")
        user_id = payload.get("sub")

        if USER_TABLE_NAME and jti and user_id:
            user_table = dynamodb.Table(USER_TABLE_NAME)
            try:
                # 1. Decrement usage count at active_tokens.<jti>
                # 2. Condition: active_tokens.<jti> must exist and be > 0
                response = user_table.update_item(
                    Key={"id": user_id},
                    UpdateExpression="SET active_tokens.#jti = active_tokens.#jti - :dec",
                    ConditionExpression="attribute_exists(active_tokens.#jti) AND active_tokens.#jti > :zero",
                    ExpressionAttributeNames={"#jti": jti},
                    ExpressionAttributeValues={":dec": 1, ":zero": 0},
                    ReturnValues="UPDATED_NEW"
                )
                
                # Update payload with remaining uses from DB
                new_vals = response.get("Attributes", {}).get("active_tokens", {})
                payload["uses_remaining"] = int(new_vals.get(jti, 0))

            except ClientError as e:
                if e.response['Error']['Code'] == 'ConditionalCheckFailedException':
                    logger.warning(f"Token {jti} invalid or exhausted.")
                    return None
                logger.error(f"DB Error validating token: {e}")
                return None
        
        return payload

    except Exception as e:
        logger.error(f"Validation error: {e}")
        return None