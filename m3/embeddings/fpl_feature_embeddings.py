"""
FPL Embeddings: feature-vector embeddings for each player-gameweek row.

- Each CSV row = 1 document (season + player + GW + fixture stats)
- Two different embedding models -> two FAISS indexes
- Helper to build/load indexes
- Helper to run semantic search given a user query + selected model
"""

import os
from typing import Dict, List, Optional, Tuple

import pandas as pd
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document


# -------------------------------------------------------------------
# 1. CONFIG
# -------------------------------------------------------------------

CSV_PATH_DEFAULT = "fpl_two_seasons.csv"
FAISS_DIR_DEFAULT = "faiss_indexes_player_gw"

# Use TWO different embedding models (requirement from milestone)
EMBEDDING_MODELS: Dict[str, str] = {
    # small, fast
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    # stronger, a bit heavier
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
}


# -------------------------------------------------------------------
# 2. ROW  ->  TEXT DOCUMENT
# -------------------------------------------------------------------

def row_to_fpl_doc(row: pd.Series) -> Tuple[str, Dict]:
    """
    Convert one CSV row into:
      - page_content: natural language description
      - metadata: structured fields (season, GW, name, teams, position, etc.)
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
    total_points = row.get("total_points", 0)
    threat = row.get("threat", 0)
    ict = row.get("ict_index", 0)
    value = row.get("value", 0)

    text_parts = [
        f"Season {season}, Gameweek {gw}.",
        f"Player: {name}, Position: {pos}.",
        f"Fixture: {home_team} vs {away_team}.",
        f"Minutes played: {minutes}.",
        f"Goals: {goals}, Assists: {assists}, Clean sheets: {cs}.",
        f"Total FPL points: {total_points}.",
        f"Threat: {threat}, ICT index: {ict}, Value: {value}.",
    ]
    page_content = " ".join(str(p) for p in text_parts if p)

    metadata = {
        "season": str(season),
        "GW": str(gw),
        "name": str(name),
        "position": str(pos),
        "home_team": str(home_team),
        "away_team": str(away_team),
        "total_points": float(total_points) if pd.notna(total_points) else None,
    }

    return page_content, metadata


# -------------------------------------------------------------------
# 3. BUILD / LOAD INDEXES
# -------------------------------------------------------------------

def build_player_gw_indexes(
    csv_path: str = CSV_PATH_DEFAULT,
    faiss_dir: str = FAISS_DIR_DEFAULT,
) -> Dict[str, FAISS]:
    """
    Build FAISS indexes for EACH embedding model in EMBEDDING_MODELS.
    Each row in the CSV becomes one Document.

    Returns: dict { model_key -> FAISS vectorstore }
    """
    os.makedirs(faiss_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    # (Optional) drop rows with missing player name
    df = df[df["name"].notna()].reset_index(drop=True)

    # Prepare all documents once (shared for all models)
    docs: List[Document] = []
    for _, row in df.iterrows():
        text, meta = row_to_fpl_doc(row)
        if not text.strip():
            continue
        docs.append(Document(page_content=text, metadata=meta))

    indexes: Dict[str, FAISS] = {}

    for key, model_name in EMBEDDING_MODELS.items():
        print(f"🔧 Building FAISS index for model '{key}' → {model_name}")
        embeddings = HuggingFaceEmbeddings(model_name=model_name)

        vs = FAISS.from_documents(docs, embeddings)
        save_path = os.path.join(faiss_dir, key)
        vs.save_local(save_path)
        indexes[key] = vs
        print(f"✅ Saved index to: {save_path}")

    return indexes


def load_player_gw_indexes(
    faiss_dir: str = FAISS_DIR_DEFAULT,
) -> Dict[str, FAISS]:
    """
    Load all FAISS indexes found in `faiss_dir` for the models in EMBEDDING_MODELS.
    """
    indexes: Dict[str, FAISS] = {}

    for key, model_name in EMBEDDING_MODELS.items():
        index_path = os.path.join(faiss_dir, key)
        if os.path.isdir(index_path):
            print(f"📂 Loading FAISS index for '{key}' from {index_path}")
            embeddings = HuggingFaceEmbeddings(model_name=model_name)
            vs = FAISS.load_local(
                index_path,
                embeddings,
                allow_dangerous_deserialization=True,
            )
            indexes[key] = vs

    return indexes


def init_player_gw_indexes(
    csv_path: str = CSV_PATH_DEFAULT,
    faiss_dir: str = FAISS_DIR_DEFAULT,
) -> Dict[str, FAISS]:
    """
    Convenience function for Streamlit:

    - If FAISS indexes already exist on disk → load them
    - Otherwise → build them from CSV and save

    Use this once in your Streamlit startup.
    """
    indexes = load_player_gw_indexes(faiss_dir)
    if indexes:
        return indexes

    # Nothing on disk yet → build them
    return build_player_gw_indexes(csv_path=csv_path, faiss_dir=faiss_dir)


# -------------------------------------------------------------------
# 4. SEMANTIC SEARCH (USER QUERY → EMBEDDINGS → TOP-10 ROWS)
# -------------------------------------------------------------------

def semantic_search_player_gw(
    query: str,
    model_key: str,
    indexes: Dict[str, FAISS],
    top_k: int = 10,
    season: Optional[str] = None,
    gw: Optional[str] = None,
):
    """
    Take a user query, choose embedding model by `model_key`,
    and return top_k most similar rows (documents).

    You can optionally filter by season and GW using metadata.
    """
    if model_key not in indexes:
        raise ValueError(
            f"Model '{model_key}' not in available indexes: {list(indexes.keys())}"
        )

    vs = indexes[model_key]

    # Get a larger candidate set first (for better filtering)
    candidates = vs.similarity_search_with_score(query, k=max(top_k * 5, 50))

    filtered = []
    for doc, score in candidates:
        meta = doc.metadata or {}

        if season is not None and meta.get("season") != str(season):
            continue
        if gw is not None and meta.get("GW") != str(gw):
            continue

        filtered.append((doc, score))
        if len(filtered) >= top_k:
            break

    # If no filters given or not enough hits, fall back to best `top_k` overall
    if not filtered:
        filtered = candidates[:top_k]

    return filtered


if __name__ == "__main__":
    # For quick testing
    indexes = init_player_gw_indexes()

    query = "Which midfielders scored the most points in gameweek 5?"
    results = semantic_search_player_gw(
        query=query,
        model_key="mpnet",
        indexes=indexes,
        top_k=5,
        gw="5",
    )

    print(f"Top results for query: {query}\n")
    for doc, score in results:
        print(f"Score: {score:.4f} | Meta: {doc.metadata}")
        print(f"Content: {doc.page_content}\n")