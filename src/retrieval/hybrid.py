"""Retrieval pipeline: BM25 + vector -> RRF fusion -> cross-encoder rerank.

Every stage is switchable through RetrievalConfig so the ablation table
(BM25 only / vector only / RRF / RRF + rerank / Weaviate hybrid) uses one code path.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Literal

import weaviate
from sentence_transformers import CrossEncoder, SentenceTransformer
from weaviate.classes.query import HybridFusion

from src.config import COLLECTION, EMBED_MODEL, QUERY_PREFIX, RERANK_MODEL
from src.retrieval.fusion import rrf_fuse

Mode = Literal["bm25", "vector", "rrf", "weaviate_hybrid"]


@dataclass
class RetrievalConfig:
    mode: Mode = "rrf"
    n_bm25: int = 50       # BM25 candidates fed to RRF
    n_vector: int = 50     # vector candidates fed to RRF
    n_fused: int = 30      # candidate pool size for every mode (keeps ablations fair)
    top_n: int = 5         # final context size
    rrf_k: int = 60
    alpha: float = 0.5     # weaviate_hybrid only: 0 = pure BM25, 1 = pure vector
    rerank: bool = True


@dataclass
class Hit:
    chunk_id: str
    title: str
    heading_path: list[str]
    source_url: str
    text: str       # raw chunk body: what the citation validator checks quotes against
    content: str    # header + body: what was indexed and what the reranker reads
    score: float = 0.0


@dataclass
class RetrievalResult:
    hits: list[Hit]                 # final top_n
    candidates: list[Hit]           # pool before reranking
    timings_ms: dict[str, float] = field(default_factory=dict)


def _hit(obj) -> Hit:
    p = obj.properties
    return Hit(
        chunk_id=p["chunk_id"],
        title=p["title"],
        heading_path=p["heading_path"],
        source_url=p["source_url"],
        text=p["text"],
        content=p["content"],
    )


class Retriever:
    def __init__(self, config: RetrievalConfig | None = None):
        self.config = config or RetrievalConfig()
        self.embedder = SentenceTransformer(EMBED_MODEL)
        self.reranker = CrossEncoder(RERANK_MODEL)
        self.client = weaviate.connect_to_local()
        self.col = self.client.collections.use(COLLECTION)

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> "Retriever":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # --- individual retrievers ------------------------------------------------
    def _bm25(self, query: str, k: int) -> list[Hit]:
        res = self.col.query.bm25(query=query, query_properties=["content"], limit=k)
        return [_hit(o) for o in res.objects]

    def _vector(self, qvec: list[float], k: int) -> list[Hit]:
        res = self.col.query.near_vector(near_vector=qvec, limit=k)
        return [_hit(o) for o in res.objects]

    def _weaviate_hybrid(self, query: str, qvec: list[float], k: int, alpha: float) -> list[Hit]:
        res = self.col.query.hybrid(
            query=query,
            vector=qvec,
            alpha=alpha,
            fusion_type=HybridFusion.RANKED,
            query_properties=["content"],
            limit=k,
        )
        return [_hit(o) for o in res.objects]

    # --- reranking --------------------------------------------------------------
    def _rerank(self, query: str, cands: list[Hit], top_n: int) -> list[Hit]:
        if not cands:
            return []
        pairs = [(query, c.content) for c in cands]
        scores = self.reranker.predict(pairs, batch_size=16, show_progress_bar=False)
        ranked = sorted(zip(cands, scores), key=lambda x: float(x[1]), reverse=True)
        return [replace(h, score=float(s)) for h, s in ranked[:top_n]]

    # --- main entry point ----------------------------------------------------------
    def retrieve(self, query: str, config: RetrievalConfig | None = None) -> RetrievalResult:
        cfg = config or self.config
        t: dict[str, float] = {}
        start = time.perf_counter()

        def lap(name: str, since: float) -> float:
            now = time.perf_counter()
            t[name] = (now - since) * 1000
            return now

        mark = time.perf_counter()
        qvec = self.embedder.encode(QUERY_PREFIX + query, normalize_embeddings=True).tolist()
        mark = lap("embed", mark)

        if cfg.mode == "bm25":
            candidates = self._bm25(query, cfg.n_fused)
            mark = lap("bm25", mark)
        elif cfg.mode == "vector":
            candidates = self._vector(qvec, cfg.n_fused)
            mark = lap("vector", mark)
        elif cfg.mode == "weaviate_hybrid":
            candidates = self._weaviate_hybrid(query, qvec, cfg.n_fused, cfg.alpha)
            mark = lap("weaviate_hybrid", mark)
        else:  # "rrf": our own fusion
            bm25_hits = self._bm25(query, cfg.n_bm25)
            mark = lap("bm25", mark)
            vec_hits = self._vector(qvec, cfg.n_vector)
            mark = lap("vector", mark)
            by_id = {h.chunk_id: h for h in bm25_hits + vec_hits}
            fused = rrf_fuse(
                [[h.chunk_id for h in bm25_hits], [h.chunk_id for h in vec_hits]],
                k=cfg.rrf_k,
            )
            candidates = [replace(by_id[cid], score=s) for cid, s in fused[: cfg.n_fused]]
            mark = lap("fuse", mark)

        if cfg.rerank:
            hits = self._rerank(query, candidates, cfg.top_n)
            mark = lap("rerank", mark)
        else:
            hits = candidates[: cfg.top_n]

        t["total"] = (time.perf_counter() - start) * 1000
        return RetrievalResult(hits=hits, candidates=candidates, timings_ms=t)


def main() -> None:
    import sys

    query = sys.argv[1] if len(sys.argv) > 1 else "How do I set resource limits for a container?"
    print("query:", query)

    setups = {
        "BM25 only": RetrievalConfig(mode="bm25", rerank=False),
        "Vector only": RetrievalConfig(mode="vector", rerank=False),
        "Hybrid (RRF)": RetrievalConfig(mode="rrf", rerank=False),
        "Hybrid + rerank": RetrievalConfig(mode="rrf", rerank=True),
    }
    with Retriever() as r:
        r.retrieve(query)  # warm-up so timings below exclude one-off model start-up
        for label, cfg in setups.items():
            res = r.retrieve(query, cfg)
            print(f"\n== {label}  ({res.timings_ms['total']:.0f} ms) ==")
            for h in res.hits:
                print(f"  {h.score:8.4f}  {h.chunk_id}")


if __name__ == "__main__":
    main()