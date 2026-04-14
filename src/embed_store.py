import json
import os
from typing import List, Dict

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import (
    CHUNKS_PATH,
    EMBEDDING_MODEL_NAME,
    FAISS_INDEX_PATH,
    METADATA_PATH
)


def load_chunks(chunks_path: str = CHUNKS_PATH) -> List[Dict]:
    with open(chunks_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_embedding_model(model_name: str = EMBEDDING_MODEL_NAME) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def create_embeddings(chunks: List[Dict], model: SentenceTransformer) -> np.ndarray:
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    return embeddings.astype("float32")


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index


def save_faiss_index(index: faiss.IndexFlatL2, index_path: str = FAISS_INDEX_PATH) -> None:
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    faiss.write_index(index, index_path)


def save_metadata(chunks: List[Dict], metadata_path: str = METADATA_PATH) -> None:
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)


def main():
    chunks = load_chunks()
    model = get_embedding_model()
    embeddings = create_embeddings(chunks, model)
    index = build_faiss_index(embeddings)
    save_faiss_index(index)
    save_metadata(chunks)
    print(f"Saved FAISS index with {len(chunks)} chunks.")


if __name__ == "__main__":
    main()