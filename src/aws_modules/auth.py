import os
import boto3
import jwt
import bcrypt
import uuid
import traceback
from datetime import datetime, timezone, timedelta
from aws_modules.api_utils import make_response
import logging

# --- Env Vars ---
USER_TABLE_NAME = os.getenv("USER_DYNAMODB_TABLE_NAME", "")
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "a-very-unsafe-default-secret")
dynamodb = boto3.resource("dynamodb")
logger = logging.getLogger()

# Default admin user credentials from environment variables
# Fall back to the specification defaults if not set
DEFAULT_ADMIN_USERNAME = os.getenv("DEFAULT_ADMIN_USERNAME", "")
DEFAULT_ADMIN_PASSWORD = os.getenv("DEFAULT_ADMIN_PASSWORD", "")


def hash_password(pw):
    """Hashes password for secure storage."""
    return bcrypt.hashpw(pw.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def check_password(pw, hashed_pw):
    """Checks a plaintext password against a stored hash."""
    if not hashed_pw:
        return False
    return bcrypt.checkpw(pw.encode("utf-8"), hashed_pw.encode("utf-8"))


def ensure_default_user():
    """
    Ensures the default admin user exists in the system.
    This function should be called during system initialization.
    """
    if not USER_TABLE_NAME:
        logger.error("User table not configured, cannot create default user")
        return False
    
    try:
        user_table = dynamodb.Table(USER_TABLE_NAME)
        
        # Check if default admin user already exists
        response = user_table.scan(
            FilterExpression=boto3.dynamodb.conditions.Attr("username").eq(DEFAULT_ADMIN_USERNAME)
        )
        
        if response.get("Items"):
            logger.info(f"Default admin user '{DEFAULT_ADMIN_USERNAME}' already exists")
            return True
        
        # Create the default admin user
        admin_id = str(uuid.uuid4())
        item = {
            "id": admin_id,
            "username": DEFAULT_ADMIN_USERNAME,
            "password_hash": hash_password(DEFAULT_ADMIN_PASSWORD),
            "roles": ["admin", "upload", "search", "download"],
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        
        user_table.put_item(Item=item)
        logger.info(f"Created default admin user '{DEFAULT_ADMIN_USERNAME}' with ID: {admin_id}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create default user: {e}\n{traceback.format_exc()}")
        return False


def create_token(user_id, roles):
    """Creates a JWT for a user."""
    # 10-hour expiry / 1000 uses
    payload = {
        "sub": user_id,
        "roles": roles,
        "exp": datetime.now(timezone.utc) + timedelta(hours=10),
        "iat": datetime.now(timezone.utc),
        "uses": 1000,
    }
    token = jwt.encode(payload, JWT_SECRET_KEY, algorithm="HS256")
    return f"bearer {token}"


def register_user(body, current_user_roles=None):
    """
    Stub for POST /users (NON-SPEC)
    Creates a new user. This should be admin-only.
    """
    # only admins can register users
    if "admin" not in (current_user_roles or []):
        return make_response(403, {"error": "Permission denied: 'admin' role required"})

    try:
        username = body["username"]
        password = body["password"]

        if not USER_TABLE_NAME:
            return make_response(500, {"error": "User table not configured"})

        user_table = dynamodb.Table(USER_TABLE_NAME)

        # TODO: Add logic to check if user already exists

        user_id = str(uuid.uuid4())
        # Default roles
        roles = body.get("roles", ["upload", "search", "download"])
        if body.get("is_admin", False):
            if "admin" not in roles:
                roles.append("admin")

        item = {
            "id": user_id,
            "username": username,
            "password_hash": hash_password(password),
            "roles": roles,
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
    """
    Handles PUT /authenticate
    Authenticates a user and returns a JWT.
    """
    try:
        username = body["user"]["name"]
        password = body["secret"]["password"]
        
        # Basic input validation for security
        if not username or not password:
            return make_response(400, {"error": "Username and password cannot be empty"})
        
        # Sanitize username - only allow alphanumeric characters and specific special chars
        if not isinstance(username, str) or not isinstance(password, str):
            return make_response(400, {"error": "Username and password must be strings"})
            
        logger.info(f"Authenticating user: {username}")

        if not USER_TABLE_NAME:
            return make_response(500, {"error": "User table not configured"})

        user_table = dynamodb.Table(USER_TABLE_NAME)

        response = user_table.scan(
            FilterExpression=boto3.dynamodb.conditions.Attr("username").eq(username)
        )
        items = response.get("Items", [])

        if not items:
            # Spec code for invalid user/pass is 401
            return make_response(401, {"error": "Invalid credentials"})

        user = items[0]

        if not check_password(password, user.get("password_hash")):
            return make_response(401, {"error": "Invalid credentials"})

        # --- Password is valid, create token ---
        token = create_token(user["id"], user.get("roles", []))
        logger.info(f"User '{username}' authenticated successfully with token {token}")

        return make_response(200, token)

    except KeyError:
        # Spec code for malformed body is 400
        return make_response(
            400,
            {
                "error": "There is missing field(s) in the "
                "AuthenticationRequest or it is formed improperly."
            },
        )
    except Exception as e:
        logger.error(f"Login error: {e}\n{traceback.format_exc()}")
        return make_response(500, {"error": str(e)})


def get_validated_user(event):
    """
    Parses and validates the JWT from the X-Authorization header.
    Returns the user payload if valid, None otherwise.
    """
    headers = {k.lower(): v for k, v in event.get("headers", {}).items()}
    auth_header = headers.get("x-authorization", "")

    if not auth_header.lower().startswith("bearer "):
        logger.info("Missing 'bearer ' prefix in X-Authorization header")
        return None

    token = auth_header.split(" ")[-1]

    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=["HS256"])

        # TODO: Implement 1000-use-limit check per Phase 2 doc

        # TODO: Check if user 'sub' (id) still exists in the user table.

        return payload
    except jwt.ExpiredSignatureError:
        logger.warning("Token expired")
        return None
    except jwt.InvalidTokenError:
        logger.warning(f"Token invalid: {traceback.format_exc()}")
        return None
