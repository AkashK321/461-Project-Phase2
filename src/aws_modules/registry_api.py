"""Registry API handler for managing artifacts, users, and authentication.

Provides the main Lambda handler function that routes requests to appropriate
artifact and user management functions.
"""

import re
from aws_modules.api_utils import make_response
from aws_modules.auth import (
    authenticate_user,
    get_validated_user,
    register_user,
    delete_user,
    get_all_users,
    update_user_roles,
)

import aws_modules.registry.context as ctx

# patchable globals expected by tests
dynamodb = ctx.dynamodb
TABLE_NAME = ctx.TABLE_NAME
BUCKET_NAME = ctx.BUCKET_NAME
USER_TABLE_NAME = ctx.USER_TABLE_NAME
JWT_SECRET_KEY = ctx.JWT_SECRET_KEY


# Core helpers
from aws_modules.registry.parsing import parse_event

# System init/reset + shared logger
from aws_modules.registry.system import initialize_system, reset_state, logger

# Split feature modules
from aws_modules.registry.artifacts.ingest import ingest_artifact
from aws_modules.registry.artifacts.search import search_by_regex, search_artifacts
from aws_modules.registry.artifacts.lineage import get_lineage_graph
from aws_modules.registry.artifacts.license_check import license_check
from aws_modules.registry.artifacts.cost import calculate_artifact_cost
from aws_modules.registry.artifacts.rating import rate_model
from aws_modules.registry.artifacts.by_name import get_artifacts_by_name
from aws_modules.registry.artifacts.crud import get_artifact, delete_artifact


def handler(event, context):
    """
    Main Lambda handler for the registry API.

    Routes incoming API Gateway events to appropriate handlers for artifacts,
    users, authentication, and system operations.

    :param event: The API Gateway event dictionary.
    :param context: The Lambda context object.
    :return: A response dictionary with status code and body.
    """
    # Initialize the system on first run (ensures default user exists)
    initialize_system()

    method, path, body, query_params = parse_event(event)

    logger.info(f"event: {event}")

    if method == "OPTIONS":
        return make_response(200, {"message": "CORS preflight successful"})

    if not TABLE_NAME or not BUCKET_NAME:
        return make_response(500, {"error": "missing env vars for table/bucket"})

    # Public Routes

    # basic health check
    if method == "GET" and path == "/health":
        return make_response(200, {"status": "ok"})

    # tracks
    if method == "GET" and path == "/tracks":
        return make_response(200, {"plannedTracks": ["Access control track"]})

    # authentication entry point
    if method == "PUT" and path == "/authenticate":
        if not USER_TABLE_NAME or not JWT_SECRET_KEY:
            return make_response(
                501, {"error": "This system does not support authentication."}
            )
        return authenticate_user(body)

    # Authentication
    user_payload = get_validated_user(event)

    # Reset (admin only but has bypass)
    if path == "/reset" and method == "DELETE":
        # Extract JTI to restore
        current_jti = user_payload.get("jti") if user_payload else None

        if not user_payload:
            # Bypass for testing
            try:
                return make_response(200, reset_state())
            except Exception as e:
                return make_response(500, {"error": str(e)})

        if "admin" not in user_payload.get("roles", []):
            return make_response(403, {"error": "Permission denied"})

        try:
            # Pass JTI to reset_state
            out = reset_state(restore_jti=current_jti)
            return make_response(200, out)
        except Exception as e:
            return make_response(500, {"error": str(e)})
    # Main Authentication Check

    if not user_payload:
        return make_response(
            403,
            {
                "error": "Authentication failed due to invalid or "
                "missing AuthenticationToken."
            },
        )

    # Authenticated Routes

    user_roles = user_payload.get("roles", [])
    user_id = user_payload.get("sub")
    logger.info(f"Authenticated user {user_id} with roles {user_roles}")

    # POST /users (Register)
    if method == "POST" and path == "/users":
        return register_user(body, user_roles)

    # GET /users (List Users)
    if method == "GET" and path == "/users":
        return get_all_users(user_roles)

    # Regex for user operations by ID
    user_op_match = re.match(r"/users/([^/]+)$", path)
    if user_op_match:
        target_id = user_op_match.group(1)

        # DELETE /users/{id}
        if method == "DELETE":
            return delete_user(target_id, user_id, user_roles)

        # PUT /users/{id} (Update Roles)
        if method == "PUT":
            new_roles = body.get("roles")
            return update_user_roles(target_id, new_roles, user_roles)

    # POST /artifact/byRegEx
    if method == "POST" and path == "/artifact/byRegEx":
        return search_by_regex(body)

    # POST /artifact/{type}
    if method == "POST" and path.startswith("/artifact/") and path.count("/") == 2:
        atype = path.split("/")[-1]
        return ingest_artifact(atype, body)

    # GET /artifact/{type}/{id}
    get_match = re.match(r"/artifacts/([^/]+)/([^/]+)", path)
    if method == "GET" and get_match and path.count("/") == 3:
        art_id = get_match.group(2)
        return get_artifact(art_id)

    # GET /artifact/model/{id}/lineage
    lineage_match = re.match(r"/artifact/model/([^/]+)/lineage", path)
    if method == "GET" and lineage_match:
        art_id = lineage_match.group(1)
        if not art_id:
            return make_response(400, {"error": "Missing artifact ID in path"})
        return get_lineage_graph(art_id)

    # POST /artifact/model/{id}/license-check
    license_check_match = re.match(r"/artifact/model/([^/]+)/license-check", path)
    if method == "POST" and license_check_match:
        art_id = license_check_match.group(1)
        return license_check(art_id, body)

    # GET /artifact/{type}/{id}/cost
    cost_match = re.match(r"/artifact/([^/]+)/([^/]+)/cost", path)
    if method == "GET" and cost_match:
        art_id = cost_match.group(2)
        if not art_id:
            return make_response(400, {"error": "Missing artifact ID in path"})
        return calculate_artifact_cost(art_id, query_params)

    # GET /artifact/model/{id}/rate
    rate_match = re.match(r"/artifact/model/([^/]+)/rate", path)
    if method == "GET" and rate_match:
        art_id = rate_match.group(1)
        if not art_id:
            return make_response(
                400,
                {
                    "error": "There is missing field(s) in the \
                                       artifact_id or it is formed improperly, \
                                       or is invalid."
                },
            )
        return rate_model(art_id)

    # GET /artifact/byName/{name}
    by_name_match = re.match(r"/artifact/byName/([^/]+)", path)
    if method == "GET" and by_name_match:
        name = by_name_match.group(1)
        if not name:
            return make_response(400, {"error": "Missing artifact name in path"})
        return get_artifacts_by_name(name)

    # POST /artifacts
    if method == "POST" and path == "/artifacts":
        try:
            return search_artifacts(body or [], query_params)
        except Exception as e:
            return make_response(400, {"error": str(e)})

    # DELETE /artifacts/{artifact_type}/{id}
    delete_match = re.match(r"/artifacts/([^/]+)/([^/]+)$", path)
    if method == "DELETE" and delete_match and path.count("/") == 3:
        artifact_type = delete_match.group(1)
        art_id = delete_match.group(2)
        return delete_artifact(artifact_type, art_id, user_roles)

    # anything else is a 404
    return make_response(404, {"error": f"Route not found: {method} {path}"})
