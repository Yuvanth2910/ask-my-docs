import pytest

from src.retrieval.fusion import rrf_fuse


def test_scores_match_formula():
    fused = dict(rrf_fuse([["a", "b"], ["b"]], k=60))
    assert fused["a"] == pytest.approx(1 / 61)
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)


def test_items_in_both_lists_beat_single_list_items():
    fused = rrf_fuse([["a", "b", "c"], ["c", "d", "a"]], k=60)
    assert {fused[0][0], fused[1][0]} == {"a", "c"}


def test_empty_input():
    assert rrf_fuse([], k=60) == []