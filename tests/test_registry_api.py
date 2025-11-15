import os
import sys
import json
import base64
import pytest

# --- Make AWS happy in tests (dummy region) ---
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")

# --- Make src/ importable so "aws_modules" works ---
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))  # repo root
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Now this matches how registry_api and its internal imports are written
from aws_modules import registry_api as reg


# ---------- helpers ----------


def decode_body(resp):
    """
    Helper for make_response-style outputs:
    expects a dict with keys: statusCode, body (JSON string).
    """
    assert isinstance(resp, dict)
    body = resp.get("body")
    if isinstance(body, str):
        return resp["statusCode"], json.loads(body)
    return resp["statusCode"], body


# ---------- parse_semver / version_satisfies ----------


def test_parse_semver_valid():
    assert reg.parse_semver("1.2.3") == (1, 2, 3)
    assert reg.parse_semver("v1.2.3") == (1, 2, 3)
    assert reg.parse_semver("1.2") == (1, 2, 0)
    assert reg.parse_semver("2") == (2, 0, 0)


def test_parse_semver_invalid():
    assert reg.parse_semver("") is None
    assert reg.parse_semver("foo") is None
    assert reg.parse_semver("1.2.3.4") is None


def test_version_satisfies_exact_and_raw_fallback():
    # exact
    assert reg.version_satisfies("1.2.3", "1.2.3")
    assert not reg.version_satisfies("1.2.3", "1.2.4")

    # raw fallback when constraint not parseable
    assert reg.version_satisfies("weird", "weird")
    assert not reg.version_satisfies("weird", "other")


def test_version_satisfies_bounded():
    # inclusive bounds
    assert reg.version_satisfies("1.2.0", "1.2.0-1.3.0")
    assert reg.version_satisfies("1.3.0", "1.2.0-1.3.0")
    assert not reg.version_satisfies("1.4.0", "1.2.0-1.3.0")


def test_version_satisfies_tilde():
    # ~1.2.0 => [1.2.0, 1.3.0)
    assert reg.version_satisfies("1.2.0", "~1.2.0")
    assert reg.version_satisfies("1.2.5", "~1.2.0")
    assert not reg.version_satisfies("1.3.0", "~1.2.0")


def test_version_satisfies_caret():
    # ^1.2.0 => [1.2.0, 2.0.0)
    assert reg.version_satisfies("1.2.0", "^1.2.0")
    assert reg.version_satisfies("1.9.9", "^1.2.0")
    assert not reg.version_satisfies("2.0.0", "^1.2.0")

    # ^0.2.3 => [0.2.3, 0.3.0)
    assert reg.version_satisfies("0.2.3", "^0.2.3")
    assert reg.version_satisfies("0.2.9", "^0.2.3")
    assert not reg.version_satisfies("0.3.0", "^0.2.3")

    # ^0.0.3 => [0.0.3, 0.0.4)
    assert reg.version_satisfies("0.0.3", "^0.0.3")
    assert not reg.version_satisfies("0.0.4", "^0.0.3")


# ---------- parse_event ----------


def test_parse_event_plain_json():
    event = {
        "rawPath": "/artifacts",
        "requestContext": {"http": {"method": "POST"}},
        "body": json.dumps({"name": "bert"}),
        "isBase64Encoded": False,
    }
    method, path, body, _ = reg.parse_event(event)
    assert method == "POST"
    assert path == "/artifacts"
    assert body == {"name": "bert"}


def test_parse_event_base64():
    payload = {"types": ["hf"]}
    encoded = base64.b64encode(json.dumps(payload).encode("utf-8")).decode("utf-8")
    event = {
        "rawPath": "/artifacts",
        "requestContext": {"http": {"method": "POST"}},
        "body": encoded,
        "isBase64Encoded": True,
    }
    method, path, body, _ = reg.parse_event(event)
    assert method == "POST"
    assert path == "/artifacts"
    assert body == payload


# ---------- search_artifacts (with fake Dynamo) ----------


class FakeTable:
    def __init__(self, items):
        self._items = items

    def scan(self):
        return {"Items": self._items}


class FakeDynamo:
    def __init__(self, items):
        self._items = items

    def Table(self, name):
        return FakeTable(self._items)


def _get_items_from_response(resp):
    status, body = decode_body(resp)
    assert status == 200
    return body


def test_search_artifacts_pagination(monkeypatch):
    items = [
        {"id": "1", "model_name": "a", "version": "1.0.0", "type": "hf"},
        {"id": "2", "model_name": "b", "version": "1.0.0", "type": "hf"},
        {"id": "3", "model_name": "c", "version": "1.0.0", "type": "hf"},
    ]

    monkeypatch.setattr(reg, "dynamodb", FakeDynamo(items))
    reg.TABLE_NAME = "dummy"

    resp_page1 = reg.search_artifacts([{"name": "*"}], {"offset": "1"})
    status1, body1 = decode_body(resp_page1)
    assert status1 == 200
    assert len(body1) == 3
    assert resp_page1["headers"].get("Offset") is None

    resp_page2 = reg.search_artifacts([{"name": "*"}], {"offset": "2"})
    status2, body2 = decode_body(resp_page2)
    assert status2 == 200
    assert len(body2) == 0


def test_search_artifacts_type_and_name_regex(monkeypatch):
    items = [
        {"id": "1", "model_name": "alpha-bert", "version": "1.0.0", "type": "hf"},
        {"id": "2", "model_name": "gamma-gpt", "version": "1.0.0", "type": "local"},
        {"id": "3", "model_name": "beta-bert", "version": "1.2.0", "type": "hf"},
    ]

    monkeypatch.setattr(reg, "dynamodb", FakeDynamo(items))
    reg.TABLE_NAME = "dummy"

    resp = reg.search_artifacts([{"types": ["hf"], "name": "bert"}], {})
    res_items = _get_items_from_response(resp)
    ids = {it["id"] for it in res_items}
    assert ids == {"1", "3"}


def test_search_artifacts_bad_regex_falls_back_to_substring(monkeypatch):
    items = [
        {"id": "1", "model_name": "my-model", "version": "1.0.0", "type": "hf"},
        {"id": "2", "model_name": "other", "version": "1.0.0", "type": "hf"},
    ]

    monkeypatch.setattr(reg, "dynamodb", FakeDynamo(items))
    reg.TABLE_NAME = "dummy"

    # '(' is invalid regex -> should fallback to substring search
    resp = reg.search_artifacts([{"name": "("}], {})
    res_items = _get_items_from_response(resp)
    # neither name contains '(' so result is empty
    assert res_items == []


def test_search_artifacts_version_filters(monkeypatch):
    items = [
        {"id": "1", "model_name": "m", "version": "1.0.0", "type": "hf"},
        {"id": "2", "model_name": "m", "version": "1.2.0", "type": "hf"},
        {"id": "3", "model_name": "m", "version": "1.5.0", "type": "hf"},
        {"id": "4", "model_name": "m", "version": "2.0.0", "type": "hf"},
    ]

    monkeypatch.setattr(reg, "dynamodb", FakeDynamo(items))
    reg.TABLE_NAME = "dummy"

    # bounded
    resp = reg.search_artifacts([{"name": "m", "version": "1.2.0-1.5.0"}], {})
    ids = {it["id"] for it in _get_items_from_response(resp)}
    assert ids == {"2", "3"}

    # tilde
    resp2 = reg.search_artifacts([{"version_range": "~1.2.0"}], {})
    ids2 = {it["id"] for it in _get_items_from_response(resp2)}
    assert ids2 == {"2"}

    # caret
    resp3 = reg.search_artifacts([{"version": "^1.0.0"}], {})
    ids3 = {it["id"] for it in _get_items_from_response(resp3)}
    assert ids3 == {"1", "2", "3"}


# ---------- ingest_artifact (happy path, everything mocked) ----------


def test_ingest_artifact_happy_path(monkeypatch, tmp_path):
    # make env sane
    reg.TABLE_NAME = "table"
    reg.BUCKET_NAME = "bucket"

    urls_seen = []

    def fake_classify_url(url):
        urls_seen.append(url)
        return "model"

    def fake_get_repo_id(url, url_type):
        return "org/my-model"

    def fake_get_base_model_from_card(repo):
        return "base/model", "inferred", "none"

    tmp_zip_file = tmp_path / "model.zip"

    def fake_snapshot_download(repo_id, local_dir):
        # create the directory so shutil.make_archive won't complain
        os.makedirs(local_dir, exist_ok=True)

    def fake_make_archive(base_name, fmt, root_dir):
        tmp_zip_file.write_bytes(b"dummy")
        return str(tmp_zip_file)

    def fake_upload_model(path, s3_key):
        assert s3_key.startswith("models/")
        return True

    saved_metadata = {"id": "model-id"}

    def fake_save_model_metadata(name, version, s3_key, scores):
        return saved_metadata

    # Fake Dynamo Table for update_item
    class FakeTable2:
        def update_item(self, **kwargs):
            # Just ensure the key is what we expect
            assert kwargs["Key"]["id"] == "model-id"
            return {}

    class FakeDynamo2:
        def Table(self, name):
            return FakeTable2()

    # monkeypatch all externals used by ingest_artifact
    monkeypatch.setattr(reg, "classify_url", fake_classify_url)
    monkeypatch.setattr(reg, "get_repo_id", fake_get_repo_id)
    monkeypatch.setattr(reg, "get_base_model_from_card", fake_get_base_model_from_card)
    monkeypatch.setattr(reg, "snapshot_download", fake_snapshot_download)
    monkeypatch.setattr(reg.shutil, "make_archive", fake_make_archive)
    monkeypatch.setattr(reg, "upload_model", fake_upload_model)
    monkeypatch.setattr(reg, "save_model_metadata", fake_save_model_metadata)
    monkeypatch.setattr(reg, "dynamodb", FakeDynamo2())

    payload = {"url": "https://huggingface.co/org/my-model"}
    resp = reg.ingest_artifact("hf", payload)
    status, body = decode_body(resp)
    assert status == 201
    assert body["id"] == "model-id"
    assert "s3_key" in body
    assert "model" in body
    assert urls_seen == ["https://huggingface.co/org/my-model"]


# ---------- handler tests ----------


@pytest.fixture(autouse=True)
def _set_env_basics(monkeypatch):
    # Ensure env-related globals are non-empty for handler tests by default
    reg.TABLE_NAME = "table"
    reg.BUCKET_NAME = "bucket"
    reg.USER_TABLE_NAME = "user-table"
    reg.JWT_SECRET_KEY = "secret"
    yield


def test_handler_health_ok():
    event = {
        "rawPath": "/health",
        "requestContext": {"http": {"method": "GET"}},
    }
    resp = reg.handler(event, None)
    status, body = decode_body(resp)
    assert status == 200
    assert body["status"] == "ok"


def test_handler_tracks_ok():
    event = {
        "rawPath": "/tracks",
        "requestContext": {"http": {"method": "GET"}},
    }
    resp = reg.handler(event, None)
    status, body = decode_body(resp)
    assert status == 200
    assert body["plannedTracks"] == ["Access control track"]


def test_handler_missing_env_vars():
    # Temporarily clear to hit the 500 early-exit
    reg.TABLE_NAME = ""
    reg.BUCKET_NAME = ""
    event = {
        "rawPath": "/artifacts",
        "requestContext": {"http": {"method": "POST"}},
        "body": "{}",
    }
    resp = reg.handler(event, None)
    status, body = decode_body(resp)
    assert status == 500
    assert "missing env vars" in body["error"]


def test_handler_authenticate_calls_authenticate_user(monkeypatch):
    called = {}

    def fake_auth(body):
        called["body"] = body
        return reg.make_response(200, {"token": "abc"})

    monkeypatch.setattr(reg, "authenticate_user", fake_auth)

    event = {
        "rawPath": "/authenticate",
        "requestContext": {"http": {"method": "PUT"}},
        "body": json.dumps({"username": "u", "password": "p"}),
    }
    resp = reg.handler(event, None)
    status, body = decode_body(resp)
    assert status == 200
    assert body["token"] == "abc"
    assert called["body"] == {"username": "u", "password": "p"}


def test_handler_protected_route_auth_failure(monkeypatch):
    # make env valid again
    reg.TABLE_NAME = "table"
    reg.BUCKET_NAME = "bucket"

    def fake_get_validated_user(event):
        return None

    monkeypatch.setattr(reg, "get_validated_user", fake_get_validated_user)

    event = {
        "rawPath": "/artifacts",
        "requestContext": {"http": {"method": "POST"}},
        "body": "{}",
    }
    resp = reg.handler(event, None)
    status, body = decode_body(resp)
    assert status == 403
    assert "Authentication failed" in body["error"]


def test_handler_reset_admin_calls_reset_state(monkeypatch):
    reg.TABLE_NAME = "table"
    reg.BUCKET_NAME = "bucket"

    def fake_get_validated_user(event):
        return {"sub": "admin-user", "roles": ["admin"]}

    called = {}

    def fake_reset_state():
        called["x"] = True
        return {"reset": "ok"}

    monkeypatch.setattr(reg, "get_validated_user", fake_get_validated_user)
    monkeypatch.setattr(reg, "reset_state", fake_reset_state)

    event = {
        "rawPath": "/reset",
        "requestContext": {"http": {"method": "DELETE"}},
        "body": "{}",
    }
    resp = reg.handler(event, None)
    status, body = decode_body(resp)
    assert status == 200
    assert body["reset"] == "ok"
    assert called["x"] is True


def test_handler_users_calls_register_user(monkeypatch):
    def fake_get_validated_user(event):
        return {"sub": "u", "roles": ["admin"]}

    called = {}

    def fake_register_user(body, roles):
        called["body"] = body
        called["roles"] = roles
        return reg.make_response(201, {"id": "user-id"})

    monkeypatch.setattr(reg, "get_validated_user", fake_get_validated_user)
    monkeypatch.setattr(reg, "register_user", fake_register_user)

    event = {
        "rawPath": "/users",
        "requestContext": {"http": {"method": "POST"}},
        "body": json.dumps({"username": "new"}),
    }
    resp = reg.handler(event, None)
    status, body = decode_body(resp)
    assert status == 201
    assert body["id"] == "user-id"
    assert called["body"] == {"username": "new"}
    assert "admin" in called["roles"]


def test_handler_ingest_route_calls_ingest_artifact(monkeypatch):
    def fake_get_validated_user(event):
        return {"sub": "u", "roles": ["upload"]}

    called = {}

    def fake_ingest_artifact(art_type, payload):
        called["type"] = art_type
        called["payload"] = payload
        return reg.make_response(201, {"id": "model-id"})

    monkeypatch.setattr(reg, "get_validated_user", fake_get_validated_user)
    monkeypatch.setattr(reg, "ingest_artifact", fake_ingest_artifact)

    event = {
        "rawPath": "/artifact/hf",
        "requestContext": {"http": {"method": "POST"}},
        "body": json.dumps({"urls": ["u"]}),
    }
    resp = reg.handler(event, None)
    status, body = decode_body(resp)
    assert status == 201
    assert body["id"] == "model-id"
    assert called["type"] == "hf"
    assert called["payload"] == {"urls": ["u"]}


def test_handler_get_artifact_found_and_not_found(monkeypatch):
    def fake_get_validated_user(event):
        return {"sub": "u", "roles": ["download"]}

    def fake_get_model_by_id(mid):
        if mid == "exists":
            # Add the fields your API handler expects
            return {
                "id": "exists",
                "model_name": "m",
                "type": "model",
                "source_url": "http://example.com",
                "s3_key": "models/exists/m.zip",  # Add s3_key for download_url
            }
        return None

    # Mock the new s3_utils function
    def fake_generate_presigned_download_url(s3_key, expiration=3600):
        return f"https://s3.signed.url/for/{s3_key}"

    monkeypatch.setattr(reg, "get_validated_user", fake_get_validated_user)
    monkeypatch.setattr(reg, "get_model_by_id", fake_get_model_by_id)
    monkeypatch.setattr(
        reg, "generate_presigned_download_url", fake_generate_presigned_download_url
    )

    # found
    event_found = {
        "rawPath": "/artifacts/model/exists",
        "requestContext": {"http": {"method": "GET"}},
        "queryStringParameters": {},  # Need to add this
    }
    resp_found = reg.handler(event_found, None)
    status_f, body_f = decode_body(resp_found)
    assert status_f == 200
    # Assert the spec-compliant structure
    assert body_f["metadata"]["id"] == "exists"
    assert body_f["metadata"]["name"] == "m"
    assert body_f["data"]["url"] == "http://example.com"
    assert "https://s3.signed.url" in body_f["data"]["download_url"]

    # not found
    event_not_found = {
        "rawPath": "/artifacts/model/nope",
        "requestContext": {"http": {"method": "GET"}},
        "queryStringParameters": {},  # Need to add this
    }
    resp_not_found = reg.handler(event_not_found, None)
    status_nf, body_nf = decode_body(resp_not_found)
    assert status_nf == 404
    assert "not exist" in body_nf["error"]


def test_handler_artifacts_calls_search_artifacts(monkeypatch):
    def fake_get_validated_user(event):
        return {"sub": "u", "roles": ["search"]}

    called = {}

    def fake_search_artifacts(query_array, query_params):
        called["payload"] = query_array[0]  # Check the first query in the array
        return reg.make_response(200, [{"id": "x"}])

    monkeypatch.setattr(reg, "get_validated_user", fake_get_validated_user)
    monkeypatch.setattr(reg, "search_artifacts", fake_search_artifacts)

    event = {
        "rawPath": "/artifacts",
        "requestContext": {"http": {"method": "POST"}},
        "body": json.dumps([{"name": "bert"}]),  # Pass a list
        "queryStringParameters": {},  # Add query params
    }
    resp = reg.handler(event, None)
    status, body = decode_body(resp)
    assert status == 200
    assert body == [{"id": "x"}]
    assert called["payload"] == {"name": "bert"}


def test_handler_unknown_route():
    reg.TABLE_NAME = "table"
    reg.BUCKET_NAME = "bucket"

    def fake_get_validated_user(event):
        return {"sub": "u", "roles": []}

    # need a valid user to get past auth check
    reg.get_validated_user = fake_get_validated_user

    event = {
        "rawPath": "/nope",
        "requestContext": {"http": {"method": "GET"}},
    }
    resp = reg.handler(event, None)
    status, body = decode_body(resp)
    assert status == 404
    assert "Route not found" in body["error"]
