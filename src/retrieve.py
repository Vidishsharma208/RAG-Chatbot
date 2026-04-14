import json
from typing import List, Dict

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import (
    EMBEDDING_MODEL_NAME,
    FAISS_INDEX_PATH,
    METADATA_PATH,
    TOP_K
)


def load_faiss_index(index_path: str = FAISS_INDEX_PATH):
    return faiss.read_index(index_path)


def load_metadata(metadata_path: str = METADATA_PATH) -> List[Dict]:
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_embedding_model(model_name: str = EMBEDDING_MODEL_NAME) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def retrieve_relevant_chunks(query: str, top_k: int = TOP_K) -> List[Dict]:
    index = load_faiss_index()
    metadata = load_metadata()
    model = get_embedding_model()

    query_embedding = model.encode([query], convert_to_numpy=True).astype("float32")
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for rank, idx in enumerate(indices[0]):
        if idx < len(metadata):
            item = metadata[idx].copy()
            item["rank"] = rank + 1
            item["distance"] = float(distances[0][rank])
            results.append(item)

    return results


if __name__ == "__main__":
    query = "Can eBay suspend a user account?"
    results = retrieve_relevant_chunks(query)

    for item in results:
        print("=" * 80)
        print(f"Rank: {item['rank']}")
        print(f"Section: {item['section']}")
        print(f"Chunk ID: {item['chunk_id']}")
        print(item["text"][:500])