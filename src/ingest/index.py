"""Embed chunks with bge-small and load them into Weaviate (self-provided vectors)."""
from __future__ import annotations

import json

import weaviate
import weaviate.classes as wvc
from sentence_transformers import SentenceTransformer
from weaviate.classes.config import Configure, DataType, Property, Tokenization
from weaviate.util import generate_uuid5

from src.config import CHUNKS_PATH, COLLECTION, EMBED_MODEL

EMBED_BATCH = 64
INSERT_BATCH = 256


def build_content(row: dict) -> str:
    """Text used for BM25 and embeddings: context header + chunk body."""
    header = row["title"]
    if row["heading_path"]:
        header += " | " + " > ".join(row["heading_path"])
    return f"{header}\n{row['text']}"


def create_collection(client: weaviate.WeaviateClient):
    if client.collections.exists(COLLECTION):
        client.collections.delete(COLLECTION)  # full rebuild each run
    return client.collections.create(
        COLLECTION,
        vector_config=Configure.Vectors.self_provided(),
        properties=[
            Property(name="chunk_id", data_type=DataType.TEXT,
                     tokenization=Tokenization.FIELD, index_searchable=False),
            Property(name="doc_slug", data_type=DataType.TEXT,
                     tokenization=Tokenization.FIELD, index_searchable=False),
            Property(name="title", data_type=DataType.TEXT, index_searchable=False),
            Property(name="heading_path", data_type=DataType.TEXT_ARRAY,
                     index_searchable=False),
            Property(name="source_url", data_type=DataType.TEXT,
                     index_searchable=False, index_filterable=False),
            # raw chunk body: what the citation validator matches quotes against
            Property(name="text", data_type=DataType.TEXT,
                     index_searchable=False, index_filterable=False),
            Property(name="n_tokens", data_type=DataType.INT),
            # header + body: the only BM25-searchable field
            Property(name="content", data_type=DataType.TEXT,
                     tokenization=Tokenization.WORD),
        ],
    )


def load_rows() -> list[dict]:
    with CHUNKS_PATH.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def main() -> None:
    rows = load_rows()
    print(f"loaded {len(rows)} chunks")

    model = SentenceTransformer(EMBED_MODEL)
    contents = [build_content(r) for r in rows]
    vectors = model.encode(
        contents,
        batch_size=EMBED_BATCH,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    objs = [
        wvc.data.DataObject(
            uuid=generate_uuid5(r["chunk_id"]),  # deterministic: re-runs are idempotent
            properties={
                "chunk_id": r["chunk_id"],
                "doc_slug": r["doc_slug"],
                "title": r["title"],
                "heading_path": r["heading_path"],
                "source_url": r["source_url"],
                "text": r["text"],
                "n_tokens": r["n_tokens"],
                "content": content,
            },
            vector=vec.tolist(),
        )
        for r, content, vec in zip(rows, contents, vectors)
    ]

    with weaviate.connect_to_local() as client:
        collection = create_collection(client)
        for i in range(0, len(objs), INSERT_BATCH):
            result = collection.data.insert_many(objs[i : i + INSERT_BATCH])
            if result.has_errors:
                print("insert errors:", list(result.errors.values())[:3])
                raise SystemExit(1)
        total = collection.aggregate.over_all(total_count=True).total_count
        print(f"indexed {total} objects into '{COLLECTION}'")


if __name__ == "__main__":
    main()