"""
FPL Feature Vector Embeddings System

This module:
1. Reads FPL CSV data
2. Converts each row to a text description
3. Creates embeddings for each row
4. Stores embeddings in FAISS database
5. Provides retrieval functionality
""" 

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os
from typing import List, Dict, Tuple, Optional


class FPLEmbeddingSystem:
    """
    System for creating and managing FPL feature embeddings using FAISS.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", embedding_dim: int = 384):
        """
        Initialize the embedding system.
        
        Args:
            model_name: Sentence transformer model name
            1. embedding_dim: Dimension of embeddings (default for all-MiniLM-L6-v2 is 384)
            2. embedding_dim: Dimension of embeddings (default for all-mpnet-base-v2 is 768)
        """
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        
        # Load model FIRST before creating folders (to avoid conflicts with existing folders)
        self.model = SentenceTransformer(model_name, device="cpu")
        
        self.index = None
        self.metadata = []  # Store original row data for each embedding
        
        # Store index files in model-specific folder (use "indices_" prefix to avoid conflicts)
        # This prevents conflicts with sentence-transformers model cache folders
        model_safe_name = model_name.replace("/", "_")
        self.model_folder = f"indices_{model_safe_name}"  # Use "indices_" prefix
        # Create folder if it doesn't exist
        if not os.path.exists(self.model_folder):
            os.makedirs(self.model_folder, exist_ok=True)
        
        # Index files inside model folder
        self.faiss_index_path = os.path.join(self.model_folder, "fpl_faiss_index.bin")
        self.metadata_path = os.path.join(self.model_folder, "fpl_metadata.pkl")
    


    def row_to_text(self, row: pd.Series) -> str:
        """
        Convert a single CSV row (one player in one fixture) into a natural-language
        description that is friendly for text embeddings.

        - Ignores stats that are 0 or NaN.
        - Only mentions saves if the player is a goalkeeper.
        - Does not assume whether the player is from the home or away team.
        """

        name = str(row.get("name", "")).strip()
        position = str(row.get("position", "")).strip()
        season = row.get("season")
        gw = row.get("GW")
        home_team = row.get("home_team")
        away_team = row.get("away_team")
        kickoff = row.get("kickoff_time")

        # Map short FPL position codes to natural words
        pos_map = {
            "GK": "goalkeeper",
            "GKP": "goalkeeper",
            "DEF": "defender",
            "MID": "midfielder",
            "FWD": "forward",
        }
        pos_upper = position.upper()
        pos_nice = pos_map.get(pos_upper, position.lower() if position else "player")

        subject = name if name else "The player"

        sentences = []

        # ── Intro sentence: season, GW, position, fixture ─────────────────────────────
        intro_parts = []

        if pd.notna(season):
            if pd.notna(gw):
                intro_parts.append(f"In Gameweek {int(gw)} of the {season} season")
            else:
                intro_parts.append(f"In the {season} season")
        elif pd.notna(gw):
            intro_parts.append(f"In Gameweek {int(gw)}")

        if intro_parts:
            intro_sentence = " ".join(intro_parts) + f", {subject} played as a {pos_nice}"
        else:
            intro_sentence = f"{subject} played as a {pos_nice}"

        if pd.notna(home_team) and pd.notna(away_team):
            intro_sentence += f" in the fixture between {home_team} and {away_team}"

        if pd.notna(kickoff):
            intro_sentence += f" on {str(kickoff)}"

        intro_sentence += "."
        sentences.append(intro_sentence)

        # ── Stats phrases (only non-zero) ─────────────────────────────────────────────
        stats_phrases = []

        def add_int_stat(col: str, template: str):
            val = row.get(col)
            if pd.notna(val) and int(val) > 0:
                n = int(val)
                stats_phrases.append(template.format(n=n))

        def add_float_stat(col: str, template: str):
            val = row.get(col)
            if pd.notna(val) and float(val) > 0:
                stats_phrases.append(template.format(v=float(val)))

        # Core stats
        minutes = row.get("minutes")
        if pd.notna(minutes) and int(minutes) > 0:
            stats_phrases.append(f"played {int(minutes)} minutes")
        else:
            stats_phrases.append(f"did not play any minutes")

        add_int_stat("goals_scored", "scored {n} goal" + ("s" if int(row.get("goals_scored", 0) or 0) != 1 else ""))
        add_int_stat("assists", "provided {n} assist" + ("s" if row.get("assists", 0) != 1 else ""))
        add_int_stat("clean_sheets", "kept {n} clean sheet" + ("s" if row.get("clean_sheets", 0) != 1 else ""))
        add_int_stat("goals_conceded", "conceded {n} goal" + ("s" if row.get("goals_conceded", 0) != 1 else ""))

        # Saves -> only if goalkeeper
        if pos_upper in ("GK", "GKP", "GOALKEEPER"):
            add_int_stat("saves", "made {n} save" + ("s" if row.get("saves", 0) != 1 else ""))
        

        add_int_stat("bonus", "earned {n} bonus point" + ("s" if row.get("bonus", 0) != 1 else ""))
        add_int_stat("total_points", "returned {n} FPL point" + ("s" if row.get("total_points", 0) != 1 else ""))

        add_int_stat("yellow_cards", "received {n} yellow card" + ("s" if row.get("yellow_cards", 0) != 1 else ""))
        add_int_stat("red_cards", "received {n} red card" + ("s" if row.get("red_cards", 0) != 1 else ""))

        # Advanced metrics (only if > 0)
        add_float_stat("ict_index", "had an ICT Index of {v:.1f}")
        add_float_stat("influence", "had an Influence score of {v:.1f}")
        add_float_stat("creativity", "had a Creativity score of {v:.1f}")
        add_float_stat("threat", "had a Threat score of {v:.1f}")
        add_float_stat("form", "had a form value of {v:.1f}")

        # Build stats sentence if we have any non-zero stats
        if stats_phrases:
            if len(stats_phrases) == 1:
                stats_sentence = f"{subject} {stats_phrases[0]}."
            else:
                stats_sentence = (
                    f"{subject} "
                    + ", ".join(stats_phrases[:-1])
                    + " and "
                    + stats_phrases[-1]
                    + "."
                )
            sentences.append(stats_sentence)
        else:
            # No meaningful stats (everything 0) → still give some text so embeddings aren't empty
            sentences.append(f"{subject} did not record any notable FPL statistics in this match.")

        return " ".join(sentences)



    def process_csv(self, csv_path: str, max_rows: Optional[int] = None) -> Tuple[List[str], List[Dict]]:
        """
        Read CSV and convert rows to text descriptions.
        
        Args:
            csv_path: Path to CSV file
            max_rows: Maximum number of rows to process (None for all)
            
        Returns:
            Tuple of (text_descriptions, metadata_list)
        """
        print(f"Reading CSV file: {csv_path}")
        df = pd.read_csv(csv_path)
        
        if max_rows:
            df = df.head(max_rows)
        
        print(f"Processing {len(df)} rows...")
        
        text_descriptions = []
        metadata_list = []
        
        for idx, row in df.iterrows():
            # Convert row to text
            text = self.row_to_text(row)
            text_descriptions.append(text)
            
            # Store original row data as metadata
            metadata_list.append(row.to_dict())
            
            if (idx + 1) % 1000 == 0:
                print(f"Processed {idx + 1}/{len(df)} rows...")
        
        print(f"Completed processing {len(text_descriptions)} rows")
        return text_descriptions, metadata_list
    
    def create_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Create embeddings for a list of texts.
        
        Args:
            texts: List of text descriptions
            batch_size: Batch size for encoding
            
        Returns:
            numpy array of embeddings (n_samples, embedding_dim)
        """
        print(f"Creating embeddings for {len(texts)} texts...")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        print(f"Created embeddings with shape: {embeddings.shape}")
        return embeddings
    
    def build_faiss_index(self, embeddings: np.ndarray, metadata: List[Dict]):
        """
        Build FAISS index from embeddings.
        
        Args:
            embeddings: numpy array of embeddings
            metadata: List of metadata dictionaries for each embedding
        """
        print(f"Building FAISS index for {len(embeddings)} embeddings...")
        
        # Convert to float32 first
        embeddings_f32 = embeddings.astype('float32')
        
        # Create FAISS index (using Inner Product for cosine similarity)
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        
        # Add embeddings to index in batches, normalizing each batch
        batch_size = 5000  # Smaller batches to avoid memory issues
        total = len(embeddings_f32)
        
        print(f"Adding embeddings in batches of {batch_size}...")
        for i in range(0, total, batch_size):
            end_idx = min(i + batch_size, total)
            batch = embeddings_f32[i:end_idx].copy()  # Copy to avoid issues
            
            # Normalize this batch manually
            norms = np.linalg.norm(batch, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            batch_normalized = batch / norms
            
            # Add normalized batch to index
            self.index.add(batch_normalized)
            
            if (i + batch_size) % 20000 == 0 or end_idx == total:
                print(f"  Added {end_idx}/{total} vectors to index...")
        
        # Store metadata
        self.metadata = metadata
        
        print(f"FAISS index built with {self.index.ntotal} vectors")
    
    def save_index(self, index_path: Optional[str] = None, metadata_path: Optional[str] = None):
        """
        Save FAISS index and metadata to disk.
        
        Args:
            index_path: Path to save FAISS index
            metadata_path: Path to save metadata
        """
        if self.index is None:
            raise ValueError("No index to save. Build index first.")
        
        index_path = index_path or self.faiss_index_path
        metadata_path = metadata_path or self.metadata_path
        
        print(f"Saving FAISS index to {index_path}...")
        faiss.write_index(self.index, index_path)
        
        print(f"Saving metadata to {metadata_path}...")
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)
        
        print("Index and metadata saved successfully!")
    
    def load_index(self, index_path: Optional[str] = None, metadata_path: Optional[str] = None):
        """
        Load FAISS index and metadata from disk.
        
        Args:
            index_path: Path to load FAISS index from
            metadata_path: Path to load metadata from
        """
        index_path = index_path or self.faiss_index_path
        metadata_path = metadata_path or self.metadata_path
        
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index file not found: {index_path}")
        
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        print(f"Loading FAISS index from {index_path}...")
        self.index = faiss.read_index(index_path)
        
        print(f"Loading metadata from {metadata_path}...")
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)
        
        print(f"Loaded index with {self.index.ntotal} vectors")
    
    def retrieve(self, query_text: str, k: int = 10) -> List[Dict]:
        """
        Retrieve similar rows based on query text.
        
        Args:
            query_text: Query text to search for
            k: Number of results to return
            
        Returns:
            List of dictionaries containing metadata and similarity scores
        """
        if self.index is None:
            raise ValueError("No index loaded. Load or build index first.")
        
        # Create embedding for query
        query_embedding = self.model.encode([query_text], convert_to_numpy=True)
        
        # Normalize for cosine similarity (manual normalization)
        query_embedding_f32 = query_embedding.astype('float32')
        norm = np.linalg.norm(query_embedding_f32, axis=1, keepdims=True)
        if norm[0, 0] > 0:
            query_embedding_normalized = query_embedding_f32 / norm
        else:
            query_embedding_normalized = query_embedding_f32
        
        # Search in FAISS index
        distances, indices = self.index.search(query_embedding_normalized, k)
        
        # Build results
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.metadata):
                result = {
                    'rank': i + 1,
                    'similarity_score': float(distance),
                    'metadata': self.metadata[idx]
                }
                results.append(result)
        
        return results
    
    def retrieve_by_example(self, example_row: Dict, k: int = 10) -> List[Dict]:
        """
        Retrieve similar rows based on an example row dictionary.
        
        Args:
            example_row: Dictionary representing a row (e.g., from CSV)
            k: Number of results to return
            
        Returns:
            List of dictionaries containing metadata and similarity scores
        """
        # Convert example row to text
        example_text = self.row_to_text(pd.Series(example_row))
        return self.retrieve(example_text, k)
    


def ensure_index(system: FPLEmbeddingSystem, csv_path: str, texts=None, metadata=None):
    """
    Load existing FAISS index+metadata if present; otherwise build and save it.
    Reuses `texts`/`metadata` if provided to avoid processing CSV twice.
    """
    if os.path.exists(system.faiss_index_path) and os.path.exists(system.metadata_path):
        print(f"\n✅ [{system.model_name}] Found existing index. Loading...")
        try:
            system.load_index()
            print(f"✅ [{system.model_name}] Index loaded successfully!")
            return texts, metadata
        except Exception as e:
            print(f"❌ [{system.model_name}] Error loading index: {e}")
            print(f"🔄 [{system.model_name}] Rebuilding index...")
            if os.path.exists(system.faiss_index_path):
                os.remove(system.faiss_index_path)
            if os.path.exists(system.metadata_path):
                os.remove(system.metadata_path)

    # Build index
    print(f"\n📊 [{system.model_name}] Building index from CSV...")
    if texts is None or metadata is None:
        texts, metadata = system.process_csv(csv_path, max_rows=None)

    embeddings = system.create_embeddings(texts)
    system.build_faiss_index(embeddings, metadata)
    system.save_index()
    return texts, metadata


def print_results(model_label: str, results: List[Dict], top_k: int = 5):
    print(f"\n--- {model_label} Top {min(top_k, len(results))} ---")
    for r in results[:top_k]:
        md = r["metadata"]
        print(f"  Rank {r['rank']} (Sim: {r['similarity_score']:.4f}) | "
              f"{md.get('season', 'N/A')} | {md.get('name', 'N/A')} | "
              f"Pos:{md.get('position', 'N/A')} | "
              f"Pts:{md.get('total_points', 'N/A')} "
              f"G:{md.get('goals_scored', 'N/A')} A:{md.get('assists', 'N/A')}")


def main():
    print("=" * 80)
    print("FPL Embedding System - Build/Load MiniLM + MPNet Indices")
    print("=" * 80)

    csv_path = "fpl_two_seasons.csv"

    # Create two systems (two models => two separate FAISS indices on disk)
    mini = FPLEmbeddingSystem(model_name="all-MiniLM-L6-v2", embedding_dim=384)
    mpnet = FPLEmbeddingSystem(model_name="all-mpnet-base-v2", embedding_dim=768)

    # Build/load MiniLM first, and reuse processed texts/metadata for MPNet
    texts, metadata = ensure_index(mini, csv_path, texts=None, metadata=None)
    _, _ = ensure_index(mpnet, csv_path, texts=texts, metadata=metadata)

    print("\n" + "=" * 80)
    print("Testing Retrieval Queries (MiniLM vs MPNet)")
    print("=" * 80)

    test_queries = [
        "Season:2022-23, name:Mohamed Salah, pos:MID, points:20, goals:2, assists:1",
        "Season:2022-23, pos:FWD, points:15, goals:1",
        "Season:2021-22, pos:DEF, clean_sheets:1, points:6",
        "name:Erling Haaland, goals:3, points:17",
        "gameweek:10, points:10",
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*80}")
        print(f"Query {i}: {query}")
        print(f"{'='*80}")

        try:
            res_mini = mini.retrieve(query, k=5)
            res_mpnet = mpnet.retrieve(query, k=5)

            print_results("MiniLM (384)", res_mini, top_k=5)
            print_results("MPNet (768)", res_mpnet, top_k=5)

        except Exception as e:
            print(f"❌ Error: {e}")

    print("\n✅ Done. You now have two indices saved at:")
    print(f"   MiniLM: {os.path.abspath(mini.faiss_index_path)}")
    print(f"   MPNet:  {os.path.abspath(mpnet.faiss_index_path)}")


if __name__ == "__main__":
    main()
