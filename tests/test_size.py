import types
from unittest.mock import patch, MagicMock
from src.scorer.metrics import size


def test_score_for_hardware_under_limit():
    total_bytes = 100000000  # 100MB
    score = size._score_on_hardware(total_bytes, "aws_server")
    assert 0 < score <= 1
    assert score > 0.5


def test_score_for_hardware_over_limit():
    total_bytes = 10_000_000_000  # 10GB
    score = size._score_on_hardware(total_bytes, "raspberry_pi")
    assert 0 <= score < 1
    assert score < 0.5


@patch("src.scorer.metrics.size.get_repo_id", return_value="mock/repo")
def test_get_size_score_model(mock_get_repo_id):
    # Mock HF_API.model_info
    fake_file = types.SimpleNamespace(size=100_000_000)
    fake_info = types.SimpleNamespace(siblings=[fake_file, fake_file])
    with patch.object(size.HF_API, "model_info", return_value=fake_info):
        scores, latency = size.get_size_score("fake_url", "model")

    assert isinstance(scores, dict)
    assert all(hw in scores for hw in size.HARDWARE_LIMITS)
    assert all(0 <= v <= 1 for v in scores.values())
    assert isinstance(latency, int)


@patch("src.scorer.metrics.size.get_repo_id", return_value="mock/repo")
def test_get_size_score_dataset(mock_get_repo_id):
    fake_file = types.SimpleNamespace(size=50_000_000)
    fake_info = types.SimpleNamespace(siblings=[fake_file])
    with patch.object(size.HF_API, "dataset_info", return_value=fake_info):
        scores, latency = size.get_size_score("fake_url", "dataset")

    assert isinstance(scores, dict)
    assert "raspberry_pi" in scores


@patch("src.scorer.metrics.size.get_repo_id", return_value="mock/repo")
def test_get_size_score_code(mock_get_repo_id):
    fake_response = {"size": 1024}  # GitHub "size" is KB
    with patch(
        "src.scorer.metrics.size.requests.get",
        return_value=MagicMock(json=lambda: fake_response),
    ):
        scores, latency = size.get_size_score("fake_url", "code")

    assert isinstance(scores, dict)
    assert "aws_server" in scores


def test_looks_like_weight_file_positive():
    assert size._looks_like_weight_file("model.safetensors")
    assert size._looks_like_weight_file("weights.pt")
    assert size._looks_like_weight_file("export.onnx")


def test_looks_like_weight_file_negative():
    assert not size._looks_like_weight_file("optimizer.pt")
    assert not size._looks_like_weight_file("training_state.bin")
    assert not size._looks_like_weight_file("notes.txt")


def test_family_key_sharded_and_nonsharded():
    # Sharded file
    key1 = size._family_key("pytorch_model-00001-of-00005.safetensors")
    assert key1 == "pytorch_model.safetensors"

    # Non-sharded file
    key2 = size._family_key("folder/model.pt")
    assert key2 == "model.pt"


def test_framework_weight_mapping():
    assert size._framework_weight("x.safetensors") == "safetensors"
    assert size._framework_weight("x.pt") == "pytorch"
    assert size._framework_weight("x.pth") == "pytorch"
    assert size._framework_weight("x.onnx") == "onnx"
    assert size._framework_weight("x.tflite") == "tflite"
    assert size._framework_weight("x.pb") == "tensorflow"
    assert size._framework_weight("x.unknown") == "other"


def test_pick_min_viable_family_prefers_smallest():
    files = [
        ("big_model.safetensors", 200),
        ("small_model.safetensors", 50),
    ]
    assert size._pick_min_viable_family(files) == 50


def test_pick_min_viable_family_tie_breaker_framework():
    files = [
        ("model.safetensors", 100),
        ("model.pt", 100),
    ]
    # safetensors preferred over pytorch on tie
    assert size._pick_min_viable_family(files) == 100


def test_pick_min_viable_family_non_weight_files():
    files = [("readme.md", 10), ("script.py", 20)]
    assert size._pick_min_viable_family(files) == 30
