"""Quick look at BM25 vs vector vs hybrid results for one query."""
from __future__ import annotations

import sys

import weaviate
from sentence_transformers import SentenceTransformer
from weaviate.classes.query import MetadataQuery

from src.config import COLLECTION, EMBED_MODEL, QUERY_PREFIX


def show(label: str, response, metric: str) -> None:
    print(f"\n== {label} ==")
    for o in response.objects:
        value = getattr(o.metadata, metric)
        value = f"{value:.4f}" if value is not None else "n/a"
        print(f"  {value:>8}  {o.properties['chunk_id']}")


def main() -> None:
    query = sys.argv[1] if len(sys.argv) > 1 else "How do I set resource limits for a container?"
    print("query:", query)

    model = SentenceTransformer(EMBED_MODEL)
    qvec = model.encode(QUERY_PREFIX + query, normalize_embeddings=True).tolist()

    with weaviate.connect_to_local() as client:
        col = client.collections.use(COLLECTION)

        bm25 = col.query.bm25(
            query=query, query_properties=["content"], limit=5,
            return_metadata=MetadataQuery(score=True),
        )
        vec = col.query.near_vector(
            near_vector=qvec, limit=5,
            return_metadata=MetadataQuery(distance=True),
        )
        hyb = col.query.hybrid(
            query=query, vector=qvec, alpha=0.5, limit=5,
            query_properties=["content"],
            return_metadata=MetadataQuery(score=True),
        )

    show("BM25", bm25, "score")
    show("Vector (distance, lower = closer)", vec, "distance")
    show("Weaviate hybrid, alpha=0.5", hyb, "score")


if __name__ == "__main__":
    main()