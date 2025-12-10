"""
FPL Feature-Vector Embeddings Module

This module:
1. Reads FPL CSV data (each row = one player-gameweek record)
2. Converts each row into a natural-language description (page_content)
3. Wraps each row as a LangChain Document with rich metadata
4. Builds TWO FAISS vectorstores (one per embedding model)
5. Provides semantic search functions that:
   - Take a user query
   - Embed it using the SELECTED model
   - Return the most similar rows with scores and metadata

Intended usage:
- Called once on app startup to build/load indexes:
    indexes = init_fpl_feature_indexes()

- Then for each query:
    results = semantic_search_fpl(
        query="strong Arsenal midfielder who scores and assists",
        model_key="mpnet",
        indexes=indexes,
        top_k=5,
        season="2022-23",
        gw="5",
    )
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import pandas as pd
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# -------------------------------------------------------------------
# 1. CONFIG
# -------------------------------------------------------------------

# Default path to your FPL CSV (adjust if needed)
CSV_PATH_DEFAULT: str = "fpl_two_seasons.csv"

# Directory where FAISS vectorstores will be saved
FAISS_DIR_DEFAULT: str = "faiss_indexes_player_gw"

# At least TWO embedding models (milestone requirement)
# - "minilm" is fast and light
# - "mpnet" is larger and usually better quality
EMBEDDING_MODELS: Dict[str, str] = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
}


# -------------------------------------------------------------------
# 2. CSV ROW  ->  DOCUMENT (TEXT + METADATA)
# -------------------------------------------------------------------

def row_to_fpl_document(row: pd.Series) -> Document:
    """
    Convert one FPL CSV row into a LangChain Document.

    - page_content: natural-language description built from numerical + categorical features
    - metadata: structured fields used later for filtering (season, GW, name, position, teams, etc.)

    This follows the "feature vector embeddings" idea in the milestone:
    we construct a textual description from numeric stats, then embed it.
    """

    season = row.get("season", "")
    gw = row.get("GW", "")
    name = row.get("name", "")
    pos = row.get("position", "")

    home_team = row.get("home_team", "")
    away_team = row.get("away_team", "")

    minutes = row.get("minutes", 0)
    goals = row.get("goals_scored", 0)
    assists = row.get("assists", 0)
    cs = row.get("clean_sheets", 0)
    goals_conceded = row.get("goals_conceded", 0)

    total_points = row.get("total_points", 0)
    threat = row.get("threat", 0)
    ict = row.get("ict_index", 0)
    influence = row.get("influence", 0)
    creativity = row.get("creativity", 0)
    value = row.get("value", 0)

    bonus = row.get("bonus", 0)
    bps = row.get("bps", 0)
    saves = row.get("saves", 0)
    yellow = row.get("yellow_cards", 0)
    red = row.get("red_cards", 0)
    form = row.get("form", 0)

    # Build a natural language description from the row
    # (this is what gets embedded by the models)
    text_parts: List[str] = []

    text_parts.append(
        f"Season {season}, Gameweek {gw}. "
        f"Player: {name}, Position: {pos}."
    )

    if home_team or away_team:
        text_parts.append(f" Fixture: {home_team} vs {away_team}.")

    text_parts.append(
        f" Minutes played: {minutes}. "
        f"Goals scored: {goals}, Assists: {assists}, "
        f"Clean sheets: {cs}, Goals conceded: {goals_conceded}."
    )

    text_parts.append(
        f" Total FPL points: {total_points}, Bonus points: {bonus}, BPS: {bps}. "
        f"Threat: {threat}, ICT index: {ict}, Influence: {influence}, "
        f"Creativity: {creativity}, Form: {form}, Value: {value}."
    )

    text_parts.append(
        f" Saves: {saves}, Yellow cards: {yellow}, Red cards: {red}."
    )

    page_content = " ".join(str(part) for part in text_parts if part)

    # Metadata is kept small & structured for filtering
    metadata = {
        "season": str(season),
        "GW": str(gw),
        "name": str(name),
        "position": str(pos),
        "home_team": str(home_team),
        "away_team": str(away_team),
        "total_points": float(total_points) if pd.notna(total_points) else None,
        "goals_scored": float(goals) if pd.notna(goals) else None,
        "assists": float(assists) if pd.notna(assists) else None,
    }

    return Document(page_content=page_content, metadata=metadata)


def _load_fpl_dataframe(csv_path: str) -> pd.DataFrame:
    """
    Load the FPL CSV into a DataFrame and perform minimal cleaning.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at: {csv_path}")

    df = pd.read_csv(csv_path)

    # Drop rows with missing player name (usually invalid)
    if "name" in df.columns:
        df = df[df["name"].notna()].reset_index(drop=True)

    return df


def _build_documents_from_csv(csv_path: str) -> List[Document]:
    """
    Convert the entire CSV into a list of Documents (one per row).
    """
    df = _load_fpl_dataframe(csv_path)
    docs: List[Document] = []

    for _, row in df.iterrows():
        doc = row_to_fpl_document(row)
        if doc.page_content.strip():
            docs.append(doc)

    return docs


# -------------------------------------------------------------------
# 3. BUILD / LOAD FAISS VECTORSTORES (ONE PER MODEL)
# -------------------------------------------------------------------

def build_fpl_feature_indexes(
    csv_path: str = CSV_PATH_DEFAULT,
    faiss_dir: str = FAISS_DIR_DEFAULT,
) -> Dict[str, FAISS]:
    """
    Build FAISS vectorstores for EACH embedding model defined in EMBEDDING_MODELS.

    - Each CSV row becomes one Document
    - For each model_key (e.g., "minilm", "mpnet"), we:
        * create HuggingFaceEmbeddings
        * build a FAISS vectorstore from all Documents
        * save it under: <faiss_dir>/<model_key>

    Returns:
        Dict[str, FAISS] : mapping from model_key to its FAISS vectorstore
    """
    os.makedirs(faiss_dir, exist_ok=True)

    print(f"📂 Loading FPL data from CSV: {csv_path}")
    docs = _build_documents_from_csv(csv_path)
    print(f"✅ Built {len(docs)} Documents from CSV (1 per row).")

    indexes: Dict[str, FAISS] = {}

    for key, model_name in EMBEDDING_MODELS.items():
        print(f"\n🔧 Building FAISS index for model '{key}' → {model_name}")
        embeddings = HuggingFaceEmbeddings(model_name=model_name)

        vs = FAISS.from_documents(docs, embeddings)

        save_path = os.path.join(faiss_dir, key)
        vs.save_local(save_path)
        indexes[key] = vs

        print(f"✅ Saved FAISS index for '{key}' to: {save_path}")

    return indexes


def load_fpl_feature_indexes(
    faiss_dir: str = FAISS_DIR_DEFAULT,
) -> Dict[str, FAISS]:
    """
    Load FAISS vectorstores for all models in EMBEDDING_MODELS from disk.

    Only models whose directories exist will be loaded.

    Returns:
        Dict[str, FAISS] : mapping from model_key to its loaded FAISS vectorstore
    """
    indexes: Dict[str, FAISS] = {}

    for key, model_name in EMBEDDING_MODELS.items():
        index_path = os.path.join(faiss_dir, key)
        if not os.path.isdir(index_path):
            continue

        print(f"📂 Loading FAISS index for '{key}' from: {index_path}")
        embeddings = HuggingFaceEmbeddings(model_name=model_name)

        vs = FAISS.load_local(
            index_path,
            embeddings,
            allow_dangerous_deserialization=True,  # required in recent LangChain versions
        )
        indexes[key] = vs

    return indexes


def init_fpl_feature_indexes(
    csv_path: str = CSV_PATH_DEFAULT,
    faiss_dir: str = FAISS_DIR_DEFAULT,
) -> Dict[str, FAISS]:
    """
    Convenience function (use this in your Streamlit app):

    - If FAISS indexes already exist in `faiss_dir` → load them
    - Otherwise → build them from the CSV and save

    Returns:
        Dict[str, FAISS] : mapping from model_key to FAISS vectorstore
    """
    indexes = load_fpl_feature_indexes(faiss_dir)
    if indexes:
        print("✅ Loaded existing FPL feature-vector indexes.")
        return indexes

    print("ℹ️ No existing indexes found. Building from CSV...")
    return build_fpl_feature_indexes(csv_path=csv_path, faiss_dir=faiss_dir)


# -------------------------------------------------------------------
# 4. SEMANTIC SEARCH: QUERY → EMBEDDING → TOP-K ROWS
# -------------------------------------------------------------------

def semantic_search_fpl(
    query: str,
    model_key: str,
    indexes: Dict[str, FAISS],
    top_k: int = 10,
    season: Optional[str] = None,
    gw: Optional[str] = None,
    position: Optional[str] = None,
) -> List[Tuple[Document, float]]:
    """
    Perform semantic search in the FPL feature-vector space using a SELECTED model.

    Steps:
      1. Use the FAISS vectorstore associated with `model_key`
      2. Embed the user query using that model
      3. Perform similarity_search_with_score
      4. Optionally filter by season/GW/position using metadata
      5. Return the top_k results as (Document, score)

    Args:
        query: natural language user query (e.g. "strong Arsenal midfielder who scores and assists")
        model_key: which embedding model to use ("minilm" or "mpnet")
        indexes: dict from init_fpl_feature_indexes()
        top_k: number of results to return
        season: optional season filter (e.g. "2022-23")
        gw: optional gameweek filter (e.g. "5")
        position: optional position filter ("FWD", "MID", "DEF", "GK")

    Returns:
        List of (Document, score) tuples, sorted by similarity (higher score = more similar)
    """
    if model_key not in indexes:
        available = list(indexes.keys())
        raise ValueError(
            f"Model '{model_key}' not found in indexes. Available: {available}"
        )

    vs = indexes[model_key]

    # Get a larger candidate set first (so filtering doesn't kill everything)
    candidate_k = max(top_k * 5, 50)
    candidates = vs.similarity_search_with_score(query, k=candidate_k)

    filtered: List[Tuple[Document, float]] = []

    for doc, score in candidates:
        meta = doc.metadata or {}

        if season is not None and meta.get("season") != str(season):
            continue
        if gw is not None and meta.get("GW") != str(gw):
            continue
        if position is not None and meta.get("position") != str(position):
            continue

        filtered.append((doc, score))
        if len(filtered) >= top_k:
            break

    # If filtering removed everything, fall back to top_k unfiltered
    if not filtered:
        filtered = candidates[:top_k]

    return filtered


def compare_models_for_query(
    query: str,
    indexes: Dict[str, FAISS],
    model_keys: Optional[List[str]] = None,
    top_k: int = 5,
    season: Optional[str] = None,
    gw: Optional[str] = None,
) -> Dict[str, List[Tuple[Document, float]]]:
    """
    Helper to compare how different embedding models behave for the SAME query.

    Returns a dict:
        {
          "minilm": [(doc, score), ...],
          "mpnet":  [(doc, score), ...],
        }

    You can easily use this in your UI to show side-by-side results.
    """
    if model_keys is None:
        model_keys = list(indexes.keys())

    results: Dict[str, List[Tuple[Document, float]]] = {}

    for model_key in model_keys:
        if model_key not in indexes:
            continue
        model_results = semantic_search_fpl(
            query=query,
            model_key=model_key,
            indexes=indexes,
            top_k=top_k,
            season=season,
            gw=gw,
        )
        results[model_key] = model_results

    return results


# -------------------------------------------------------------------
# 5. QUICK CLI TEST
# -------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 80)
    print("FPL Feature-Vector Embeddings | Test Run")
    print("=" * 80)

    indexes = init_fpl_feature_indexes()

    test_query = "strong Arsenal midfielder who scores and assists"
    print(f"\nQuery: {test_query}\n")

    for model_key in EMBEDDING_MODELS.keys():
        print(f"{'-' * 80}")
        print(f"Model: {model_key}")
        print(f"{'-' * 80}")

        results = semantic_search_fpl(
            query=test_query,
            model_key=model_key,
            indexes=indexes,
            top_k=5,
        )

        for rank, (doc, score) in enumerate(results, start=1):
            meta = doc.metadata
            print(
                f"  #{rank} | score={score:.4f} | "
                f"{meta.get('season', 'N/A')} | GW {meta.get('GW', 'N/A')} | "
                f"{meta.get('name', 'N/A')} ({meta.get('position', 'N/A')}) | "
                f"points={meta.get('total_points', 'N/A')}"
            )

        print()
