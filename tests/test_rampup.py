from unittest.mock import patch, MagicMock
import os
from src.scorer.metrics.rampup import (
    get_ramp_up,
    _parse_repo_id,
    _get_repo_tree_hf,
    _ask_llm,
    _heuristic_rampup,
    _extract_json_first,
    _fetch_file_content,
)


def test_extract_json_first_valid():
    text = 'some junk {"score": 0.8, "rationale": "clear"} trailing'
    result = _extract_json_first(text)
    assert isinstance(result, dict)
    assert result["score"] == 0.8


def test_extract_json_first_invalid_json():
    text = "{not json}"
    result = _extract_json_first(text)
    assert result is None


def test_extract_json_first_no_json():
    text = "hello world"
    result = _extract_json_first(text)
    assert result is None


# Test URL/repo_id parsing
def test_parse_repo_id_hf_model():
    """Test HuggingFace model URL parsing."""
    url = "https://huggingface.co/owner/model"
    repo_id, repo_type = _parse_repo_id(url)
    assert repo_id == "owner/model"
    assert repo_type == "model"


def test_parse_repo_id_hf_dataset():
    """Test HuggingFace dataset URL parsing."""
    url = "https://huggingface.co/datasets/owner/dataset"
    repo_id, repo_type = _parse_repo_id(url)
    assert repo_id == "owner/dataset"
    assert repo_type == "dataset"


def test_parse_repo_id_hf_space():
    """Test HuggingFace space URL parsing."""
    url = "https://huggingface.co/spaces/owner/space"
    repo_id, repo_type = _parse_repo_id(url)
    assert repo_id == "owner/space"
    assert repo_type == "space"


def test_parse_repo_id_simple_string():
    """Test simple 'user/repo' string parsing."""
    url = "owner/model"
    repo_id, repo_type = _parse_repo_id(url)
    assert repo_id == "owner/model"
    assert repo_type == "model"


def test_parse_repo_id_invalid():
    """Test invalid URL returns None."""
    url = "https://invalid.com/single"
    repo_id, repo_type = _parse_repo_id(url)
    assert repo_id is None
    assert repo_type is None


def test_parse_repo_id_empty_path():
    """Test URL with no path components."""
    url = "https://huggingface.co/"
    repo_id, repo_type = _parse_repo_id(url)
    assert repo_id is None
    assert repo_type is None


# Test file tree and README fetching via HF API
@patch("src.scorer.metrics.rampup.HF_API.list_repo_files")
@patch("src.scorer.metrics.rampup._fetch_file_content")
def test_get_repo_tree_hf_success(mock_fetch, mock_list):
    """Test successful repo tree retrieval."""
    mock_list.return_value = ["file1.py", "file2.py", "README.md"]
    mock_fetch.return_value = "# README\npip install test"

    tree, readme = _get_repo_tree_hf("owner/repo", "model")

    assert "file1.py" in tree
    assert "file2.py" in tree
    assert readme == "# README\npip install test"


@patch("src.scorer.metrics.rampup.HF_API.list_repo_files")
def test_get_repo_tree_hf_skip_dirs(mock_list):
    """Test that SKIP_DIRS are excluded from tree."""
    mock_list.return_value = [
        "normal_file.py",
        ".git/config",
        "__pycache__/cache.pkl",
        "src/main.py",
    ]

    tree, readme = _get_repo_tree_hf("owner/repo", "model")

    assert "normal_file.py" in tree
    assert ".git" not in tree
    assert "__pycache__" not in tree
    assert "src/main.py" in tree


@patch("src.scorer.metrics.rampup.HF_API.list_repo_files")
def test_get_repo_tree_hf_api_error(mock_list):
    """Test graceful handling of API errors."""
    mock_list.side_effect = Exception("API error")

    tree, readme = _get_repo_tree_hf("owner/repo", "model")

    assert tree == ""
    assert readme is None


@patch("src.scorer.metrics.rampup.hf_hub_download")
def test_fetch_file_content_success(mock_download):
    """Test successful file fetch."""
    mock_download.return_value = "/tmp/README.md"

    with patch("builtins.open", create=True) as mock_open:
        mock_open.return_value.__enter__.return_value.read.return_value = "# Content"
        result = _fetch_file_content("owner/repo", "model", "README.md")

    assert result == "# Content"


@patch("src.scorer.metrics.rampup.hf_hub_download")
def test_fetch_file_content_not_found(mock_download):
    """Test file not found returns empty string."""
    mock_download.side_effect = Exception("File not found")

    result = _fetch_file_content("owner/repo", "model", "README.md")
    assert result == ""

    """Test heuristic with few patterns."""
    readme = "# Project\nThis is a project."
    tree = "file1.txt\nfile2.txt"

    score = _heuristic_rampup(readme, tree)
    assert score == 0.15  # Minimum score


def test_heuristic_rampup_empty():
    """Test heuristic with empty content."""
    score = _heuristic_rampup("", "")
    assert score == 0.15


def test_heuristic_rampup_score_range():
    """Test that heuristic score is always in [0,1] range."""
    # Test with excessive patterns
    excessive_patterns = " ".join(["pip install"] * 100)
    score = _heuristic_rampup(excessive_patterns, "")
    assert 0.0 <= score <= 1.0


def test_ask_llm_no_env_vars():
    """Test LLM with missing environment variables."""
    # Clear environment variables
    with patch.dict(os.environ, {}, clear=True):
        score = _ask_llm("README content", "file tree")
        assert score is None


@patch("src.scorer.metrics.rampup._get_repo_tree_hf")
@patch("src.scorer.metrics.rampup._ask_llm")
def test_get_ramp_up_with_llm_mock(mock_ask_llm, mock_get_tree):
    """Test get_ramp_up with mocked LLM."""
    score, latency = get_ramp_up("https://huggingface.co/mock/repo", "model")

    # Score should be a float between 0 and 1
    assert 0.0 <= score <= 1.0
    assert isinstance(latency, int)


@patch("src.scorer.metrics.rampup._get_repo_tree_hf")
@patch("src.scorer.metrics.rampup._ask_llm")
def test_get_ramp_up_parse_error_returns_zero(mock_ask_llm, mock_get_tree):
    """Test that parsing errors return score 0."""
    mock_get_tree.side_effect = Exception("Parse error")

    score, latency = get_ramp_up("https://huggingface.co/invalid", "model")
    assert score == 0.0
    assert isinstance(latency, int)


@patch("src.scorer.metrics.rampup._get_repo_tree_hf")
@patch("src.scorer.metrics.rampup._ask_llm")
def test_get_ramp_up_invalid_repo_id_returns_zero(mock_ask_llm, mock_get_tree):
    """Test invalid repo_id returns score 0."""
    score, latency = get_ramp_up("https://invalid.com/single", "model")
    assert score == 0.0
    assert isinstance(latency, int)


@patch("src.scorer.metrics.rampup._get_repo_tree_hf")
@patch("src.scorer.metrics.rampup._ask_llm")
def test_get_ramp_up_llm_success(mock_ask_llm, mock_get_tree):
    """Test successful LLM evaluation."""
    mock_get_tree.return_value = (
        "setup.py\nREADME.md\nrequirements.txt",
        "# Test README\npip install test",
    )
    mock_ask_llm.return_value = 0.85

    score, latency = get_ramp_up("https://huggingface.co/owner/repo", "model")

    # Should use LLM score
    assert score == 0.85
    assert isinstance(latency, int)


@patch("src.scorer.metrics.rampup._get_repo_tree_hf")
@patch("src.scorer.metrics.rampup._ask_llm")
def test_get_ramp_up_heuristic_fallback(mock_ask_llm, mock_get_tree):
    """Test fallback to heuristic when LLM returns None."""
    mock_get_tree.return_value = ("file1.py\nfile2.py", "# README\npip install test")
    mock_ask_llm.return_value = None

    score, latency = get_ramp_up("https://huggingface.co/owner/repo", "model")

    # Should use heuristic scoring (which includes boosts)
    assert 0.0 <= score <= 1.0
    assert isinstance(latency, int)
    # With heuristic and "pip install" pattern, should be > 0.15
    assert score > 0.15


@patch("src.scorer.metrics.rampup._get_repo_tree_hf")
@patch("src.scorer.metrics.rampup._ask_llm")
def test_get_ramp_up_llm_score_clamping(mock_ask_llm, mock_get_tree):
    """Test that scores are clamped to [0.0, 1.0] range."""
    mock_get_tree.return_value = ("files", "README")

    # Test score > 1.0 gets clamped
    mock_ask_llm.return_value = 1.5
    score, _ = get_ramp_up("https://huggingface.co/owner/repo", "model")
    assert score == 1.0

    # Test score < 0.0 gets clamped
    mock_ask_llm.return_value = -0.5
    score, _ = get_ramp_up("https://huggingface.co/owner/repo", "model")
    assert score == 0.0


@patch("src.scorer.metrics.rampup._get_repo_tree_hf")
@patch("src.scorer.metrics.rampup._ask_llm")
def test_get_ramp_up_empty_content_heuristic(mock_ask_llm, mock_get_tree):
    """Test heuristic fallback with empty README and file list."""
    mock_get_tree.return_value = ("", "")
    mock_ask_llm.return_value = None

    score, latency = get_ramp_up("https://huggingface.co/owner/repo", "model")

    # Heuristic should return minimum score (0.15) when no patterns found
    expected_score = _heuristic_rampup("", "")
    assert score == expected_score
    assert score == 0.15  # Minimum heuristic score
    assert isinstance(latency, int)


@patch("src.scorer.metrics.rampup._get_repo_tree_hf")
@patch("src.scorer.metrics.rampup._ask_llm")
def test_get_ramp_up_type_detection(mock_ask_llm, mock_get_tree):
    """Test URL type detection when url_type is not provided."""
    mock_get_tree.return_value = ("files", "README")
    mock_ask_llm.return_value = 0.7

    # Pass just the URL and let it parse the type
    score, latency = get_ramp_up("owner/repo", "")

    assert 0.0 <= score <= 1.0
    assert isinstance(latency, int)


@patch("src.scorer.metrics.rampup.requests.Session.post")
def test_ask_llm_salvage_number(mock_post, monkeypatch):
    """LLM returns non-JSON but salvageable float."""
    monkeypatch.setenv("GEN_AI_STUDIO_API_KEY", "fakekey")
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": "score is 0.7"}}]
    }
    mock_post.return_value = mock_resp

    score = _ask_llm("README", "TREE")
    assert score == 0.7
