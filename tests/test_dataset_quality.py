from pytest import approx
import types
from unittest.mock import patch
from src.scorer.metrics import dataset_quality


def test_normalize_zero_and_negative():
    assert dataset_quality.normalize(0, 1000) == 0.0
    assert dataset_quality.normalize(-5, 1000) == 0.0


def test_normalize_within_range():
    val = dataset_quality.normalize(100, 1000)
    assert 0 < val <= 1
    assert val == approx(0.67, abs=0.01)


@patch("src.scorer.metrics.dataset_quality.get_repo_id", return_value="mock/repo")
def test_get_dataset_quality_score_with_likes(mock_get_repo_id):
    fake_info = types.SimpleNamespace(downloads=50_000, likes=100)
    with patch.object(dataset_quality.HF_API, "dataset_info", return_value=fake_info):
        score, latency = dataset_quality.get_dataset_quality_score(
            "fake_url", "dataset"
        )

    assert isinstance(score, float)
    assert 0 <= score <= 1
    assert isinstance(latency, int)


@patch("src.scorer.metrics.dataset_quality.get_repo_id", return_value="mock/repo")
def test_get_dataset_quality_score_without_likes(mock_get_repo_id):
    fake_info = types.SimpleNamespace(downloads=10_000, likes=0)
    with patch.object(dataset_quality.HF_API, "dataset_info", return_value=fake_info):
        score, latency = dataset_quality.get_dataset_quality_score("fake_url", "dataset")

    # When likes = 0, function returns downloads_score only
    assert isinstance(score, float)
    assert 0 <= score <= 1
    assert isinstance(latency, int)


def test_get_dataset_quality_score_wrong_type():
    score, latency = dataset_quality.get_dataset_quality_score("fake_url", "model")
    assert score == 0.0
    assert isinstance(latency, int)


@patch(
    "src.scorer.metrics.dataset_quality.get_repo_id", side_effect=Exception("bad repo")
)
def test_get_dataset_quality_score_repo_id_failure(mock_get_repo_id):
    score, latency = dataset_quality.get_dataset_quality_score("fake_url", "dataset")
    assert score is None
    assert isinstance(latency, int)
