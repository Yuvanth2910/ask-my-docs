"""Reciprocal Rank Fusion. Pure functions, no heavy imports, easy to unit test."""
from __future__ import annotations


def rrf_fuse(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Fuse several ranked lists of ids. score(id) = sum over lists of 1 / (k + rank).

    Ranks start at 1. Returns (id, score) sorted by score, highest first.
    Needs no score normalisation, which is why it suits BM25 + cosine fusion.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item_id in enumerate(ranking, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)