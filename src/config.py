"""Shared constants so ingest, retrieval and eval stay consistent."""
from pathlib import Path

CHUNKS_PATH = Path("data/processed/chunks.jsonl")
COLLECTION = "K8sDocs"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
# bge models expect this prefix on queries (not on documents)
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "