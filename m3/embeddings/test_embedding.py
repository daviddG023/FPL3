"""
Quick test script for embedding system with small sample
"""

from embedding import FPLEmbeddingSystem
import os

def embed_query(query: str, model_name: str, embedding_dim: int, top_k: int = 5):
    """Test with a small sample first"""
    import os
    
    print("=" * 80)
    print("Testing FPL Embedding System (Small Sample)")
    print("=" * 80)
    
    embedding_system = FPLEmbeddingSystem(model_name=model_name, embedding_dim=embedding_dim)
    # embedding_system = FPLEmbeddingSystem(model_name="all-MiniLM-L6-v2", embedding_dim=384)
    #embedding_system = FPLEmbeddingSystem(model_name="all-mpnet-base-v2", embedding_dim=768)
    
    csv_path = "fpl_two_seasons.csv"
    
    # # Check if index already exists in model folder
    # print(f"\n📁 Model folder: {embedding_system.model_folder}")
    # print(f"📄 Index file: {embedding_system.faiss_index_path}")
    # print(f"📄 Metadata file: {embedding_system.metadata_path}")
    
    if os.path.exists(embedding_system.faiss_index_path) and os.path.exists(embedding_system.metadata_path):
        print("\n✅ Found existing index. Loading...")
        try:
            embedding_system.load_index()
            print(f"✅ Loaded index with {embedding_system.index.ntotal} vectors")
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
        # Process full CSV dataset
        print("\n📊 Processing full CSV dataset...")
        texts, metadata = embedding_system.process_csv(csv_path)
        
        print(f"\nSample text descriptions:")
        for i in range(min(3, len(texts))):
            print(f"  {i+1}. {texts[i][:150]}...")
        
        # Create embeddings
        print("\n🔢 Creating embeddings...")
        embeddings = embedding_system.create_embeddings(texts)
        
        # Build FAISS index
        print("\n📚 Building FAISS index...")
        embedding_system.build_faiss_index(embeddings, metadata)
        
        # Save index for future use
        print("\n💾 Saving index to disk...")
        embedding_system.save_index()
        print("✅ Index saved successfully!")
    
    # Test retrieval
    print("\n🔍 Testing retrieval...")
    
    results = embedding_system.retrieve(query, k=top_k)
    
    print(f"\nQuery: {query}")
    print(f"\nTop {len(results)} results:")
    for result in results:
        meta = result['metadata']
        # print(meta)
        print(f"\n  Rank {result['rank']} (Score: {result['similarity_score']:.4f}):")
        print(f"    Season: {meta.get('season')} - Player: {meta.get('name')} (position: {meta.get('position')})")
        print(f"    Points: {meta.get('total_points')}, Goals: {meta.get('goals_scored')}, Assists: {meta.get('assists')}")
        print(f"    Gameweek: {meta.get('GW')}, Fixture: {meta.get('fixture')}")
    
    print("\n✅ Test completed successfully!")
    return results
if __name__ == "__main__":
    embed_query(query="Season:2021-22, name:Aaron Connolly, pos:FWD, points:1", model_name="all-MiniLM-L6-v2", embedding_dim=384, top_k=5)
    # embed_query(query="Season:2021-22, name:Aaron Connolly, pos:FWD, points:1", model_name="all-mpnet-base-v2", embedding_dim=768, top_k=5)

