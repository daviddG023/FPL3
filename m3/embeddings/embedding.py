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
import json


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
        self.model = SentenceTransformer(model_name)
        
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
        Convert a CSV row to a text description.
        
        Args:
            row: pandas Series representing a row
            
        Returns:
            Text description string
        """
        # Build text description from row
        parts = []
        
        # Add key fields
        if pd.notna(row.get('season')):
            parts.append(f"Season:{row['season']}")
        
        if pd.notna(row.get('name')):
            parts.append(f"name:{row['name']}")
        
        if pd.notna(row.get('position')):
            parts.append(f"pos:{row['position']}")
        
        # Add performance stats
        if pd.notna(row.get('total_points')):
            parts.append(f"points:{row['total_points']}")
        
        if pd.notna(row.get('goals_scored')):
            parts.append(f"goals:{row['goals_scored']}")
        
        if pd.notna(row.get('assists')):
            parts.append(f"assists:{row['assists']}")
        
        if pd.notna(row.get('minutes')):
            parts.append(f"minutes:{row['minutes']}")
        
        if pd.notna(row.get('clean_sheets')):
            parts.append(f"clean_sheets:{row['clean_sheets']}")
        
        if pd.notna(row.get('bonus')):
            parts.append(f"bonus:{row['bonus']}")
        
        if pd.notna(row.get('bps')):
            parts.append(f"bps:{row['bps']}")
        
        # Add ICT stats
        if pd.notna(row.get('ict_index')):
            parts.append(f"ict_index:{row['ict_index']}")
        
        if pd.notna(row.get('influence')):
            parts.append(f"influence:{row['influence']}")
        
        if pd.notna(row.get('creativity')):
            parts.append(f"creativity:{row['creativity']}")
        
        if pd.notna(row.get('threat')):
            parts.append(f"threat:{row['threat']}")
        
        if pd.notna(row.get('form')):
            parts.append(f"form:{row['form']}")
        
        # Add fixture info
        if pd.notna(row.get('fixture')):
            parts.append(f"fixture:{row['fixture']}")
        
        if pd.notna(row.get('GW')):
            parts.append(f"gameweek:{row['GW']}")
        
        if pd.notna(row.get('home_team')):
            parts.append(f"home_team:{row['home_team']}")
        
        if pd.notna(row.get('away_team')):
            parts.append(f"away_team:{row['away_team']}")
        
        if pd.notna(row.get('team_a_score')):
            parts.append(f"away_score:{row['team_a_score']}")
        
        if pd.notna(row.get('team_h_score')):
            parts.append(f"home_score:{row['team_h_score']}")
        
        # Add other relevant stats
        if pd.notna(row.get('goals_conceded')):
            parts.append(f"goals_conceded:{row['goals_conceded']}")
        
        if pd.notna(row.get('saves')):
            parts.append(f"saves:{row['saves']}")
        
        if pd.notna(row.get('yellow_cards')):
            parts.append(f"yellow_cards:{row['yellow_cards']}")
        
        if pd.notna(row.get('red_cards')):
            parts.append(f"red_cards:{row['red_cards']}")
        
        if pd.notna(row.get('value')):
            parts.append(f"value:{row['value']}")
        
        # Join all parts with commas
        return ", ".join(parts)
    
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


def main():
    """Main function to test the embedding system"""
    
    # Initialize system
    print("=" * 80)
    print("FPL Embedding System - Full Dataset Processing")
    print("=" * 80)
    
    embedding_system = FPLEmbeddingSystem()
    
    # Path to CSV file
    csv_path = "fpl_two_seasons.csv"
    
    # Check if index already exists
    if os.path.exists(embedding_system.faiss_index_path) and os.path.exists(embedding_system.metadata_path):
        print("\n✅ Found existing index. Loading...")
        try:
            embedding_system.load_index()
            print("✅ Index loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading index: {e}")
            print("Rebuilding index...")
            # Fall through to rebuild
            if os.path.exists(embedding_system.faiss_index_path):
                os.remove(embedding_system.faiss_index_path)
            if os.path.exists(embedding_system.metadata_path):
                os.remove(embedding_system.metadata_path)
    
    # Build index if it doesn't exist or loading failed
    if embedding_system.index is None:
        print("\n📊 Processing CSV and building index...")
        
        # Process full CSV dataset
        texts, metadata = embedding_system.process_csv(csv_path, max_rows=None)
        
        # Create embeddings
        embeddings = embedding_system.create_embeddings(texts)
        
        # Build FAISS index
        embedding_system.build_faiss_index(embeddings, metadata)
        
        # Save index
        embedding_system.save_index()
    
    # Test retrieval queries
    print("\n" + "=" * 80)
    print("Testing Retrieval Queries")
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
            results = embedding_system.retrieve(query, k=5)
            
            print(f"\nTop {len(results)} results:")
            for result in results:
                metadata = result['metadata']
                print(f"\n  Rank {result['rank']} (Similarity: {result['similarity_score']:.4f}):")
                print(f"    Season: {metadata.get('season', 'N/A')}")
                print(f"    Player: {metadata.get('name', 'N/A')}")
                print(f"    Position: {metadata.get('position', 'N/A')}")
                print(f"    Points: {metadata.get('total_points', 'N/A')}, Goals: {metadata.get('goals_scored', 'N/A')}, Assists: {metadata.get('assists', 'N/A')}")
                print(f"    Gameweek: {metadata.get('GW', 'N/A')}, Fixture: {metadata.get('fixture', 'N/A')}")
                if metadata.get('home_team') and metadata.get('away_team'):
                    print(f"    Match: {metadata.get('home_team')} vs {metadata.get('away_team')}")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Test retrieval by example
    print("\n" + "=" * 80)
    print("Testing Retrieval by Example")
    print("=" * 80)
    
    example_row = {
        'season': '2022-23',
        'name': 'Mohamed Salah',
        'position': 'MID',
        'total_points': 20,
        'goals_scored': 2,
        'assists': 1,
        'GW': 5
    }
    
    print(f"\nExample row: {example_row}")
    try:
        results = embedding_system.retrieve_by_example(example_row, k=5)
        
        print(f"\nTop {len(results)} similar results:")
        for result in results:
            metadata = result['metadata']
            print(f"\n  Rank {result['rank']} (Similarity: {result['similarity_score']:.4f}):")
            print(f"    {metadata.get('season', 'N/A')} - {metadata.get('name', 'N/A')} ({metadata.get('position', 'N/A')})")
            print(f"    Points: {metadata.get('total_points', 'N/A')}, Goals: {metadata.get('goals_scored', 'N/A')}, Assists: {metadata.get('assists', 'N/A')}")
            print(f"    Gameweek: {metadata.get('GW', 'N/A')}, Fixture: {metadata.get('fixture', 'N/A')}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n" + "=" * 80)
    print("Test completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()

