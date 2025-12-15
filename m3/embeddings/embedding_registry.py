# embeddings/embedding_registry.py
from __future__ import annotations

import os
from typing import Dict

from embeddings.embedding import FPLEmbeddingSystem

CSV_PATH_DEFAULT = "fpl_two_seasons.csv"

def _ensure_loaded(system: FPLEmbeddingSystem, csv_path: str) -> FPLEmbeddingSystem:
    """
    Load existing FAISS index+metadata if present; otherwise build and save it.
    """
    if os.path.exists(system.faiss_index_path) and os.path.exists(system.metadata_path):
        system.load_index()
        return system

    # Build index if missing
    texts, metadata = system.process_csv(csv_path, max_rows=None)
    embeddings = system.create_embeddings(texts)
    system.build_faiss_index(embeddings, metadata)
    system.save_index()
    return system


def load_embedding_systems(csv_path: str = CSV_PATH_DEFAULT) -> Dict[str, FPLEmbeddingSystem]:
    """
    Returns dict:
      {
        "minilm": FPLEmbeddingSystem(...),
        "mpnet":  FPLEmbeddingSystem(...),
      }
    Both are loaded (or built) and ready to retrieve().
    """
    minilm = FPLEmbeddingSystem(model_name="all-MiniLM-L6-v2", embedding_dim=384)
    mpnet  = FPLEmbeddingSystem(model_name="all-mpnet-base-v2", embedding_dim=768)

    _ensure_loaded(minilm, csv_path)
    _ensure_loaded(mpnet,  csv_path)

    return {"minilm": minilm, "mpnet": mpnet}
