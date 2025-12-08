"""
Test script to show Cypher query generation with entities
"""

from intent_classification_chat import IntentClassifier
from Graph_Retrieval import GraphRetrieval


def test_cypher_generation():
    """Test Cypher query generation for various queries"""
    
    classifier = IntentClassifier()
    
    # Create mock retrieval instance (without connection)
    retrieval = GraphRetrieval.__new__(GraphRetrieval)
    retrieval.query_templates = retrieval._initialize_query_templates()
    
    test_queries = [
        "How many points did Mohamed Salah score in 2022-23?",
        "Who are the top 10 defenders in the 2022-23 season?",
        "Show me arsenal fixtures for gameweek 10 in 2022-23",
        "What games are in gameweek 5 of 2022-23?",
        "Compare Mohamed Salah vs Erling Haaland in 2022-23",
        "Show me Erling Haaland's stats for gameweek 10 in 2022-23",
    ]
    test_queries_2 = [
        "How many points did Mohamed Salah score in 2022-23?",
        "Who are the best defenders to pick in GW5?",
        "Show me all fixtures for Arsenal",
        "What games are in gameweek 5?",
        "Which teams played in the 2022-23 season?",
        "Compare Mohamed Salah vs Erling Haaland this season",
        "Find players who play as defender",
        "How many total gameweeks are there?",
        "What is the highest points scored by a player?",
        "Show me all players",
    ]
    
    print("=" * 100)
    print("Cypher Query Generation Test")
    print("=" * 100)
    
    for query in test_queries_2:
        print(f"\n{'='*100}")
        print(f"Query: {query}")
        print(f"{'='*100}")
        
        # Classify intent and extract entities
        intent, metadata = classifier.classify(query)
        entities = metadata["entities"]
        
        print(f"\nIntent: {intent.value}")
        print(f"Entities Extracted: {entities}")
        
        # Generate Cypher query
        intent_str = intent.value.upper()
        query_name, cypher_query, params = retrieval.generate_cypher_query(intent_str, entities)
        
        if query_name:
            query_info = retrieval.get_query_info(query_name)
            print(f"\nSelected Query Template: {query_name}")
            print(f"Description: {query_info.get('description', 'N/A')}")
            print(f"\nGenerated Cypher Query:")
            print("-" * 100)
            print(cypher_query)
            print("-" * 100)
            print(f"\nParameters: {params}")
        else:
            print("\n❌ No suitable query template found!")
            print("This means:")
            print("  - Either the intent doesn't match any query templates")
            print("  - Or required entities are missing")
            
            # Show what queries are available for this intent
            available_queries = [
                name for name, template in retrieval.query_templates.items()
                if template["intent"] == intent_str
            ]
            if available_queries:
                print(f"\nAvailable queries for intent '{intent_str}':")
                for qname in available_queries:
                    qinfo = retrieval.get_query_info(qname)
                    print(f"  - {qname}: {qinfo.get('description', 'N/A')}")
                    print(f"    Required entities: {qinfo.get('required_entities', [])}")
                    print(f"    Optional entities: {qinfo.get('optional_entities', [])}")


if __name__ == "__main__":
    test_cypher_generation()

