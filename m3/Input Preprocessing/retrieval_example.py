"""
Example: Using Intent Classification with Graph Retrieval

This script demonstrates how to:
1. Classify user queries using IntentClassifier
2. Select and parameterize Cypher queries using GraphRetrieval
3. Execute queries and display results

This is the baseline retrieval approach (without embeddings).
"""

import sys
from intent_classification_chat import IntentClassifier, Intent
from Graph_Retrieval import GraphRetrieval


def load_config(path="m2/config.txt"):
    """Load Neo4j configuration from config file"""
    config = {}
    try:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if "=" in line:
                    key, value = line.split("=", 1)
                    config[key] = value.strip().replace('"', '')
        return config
    except FileNotFoundError:
        print(f"Config file not found: {path}")
        return {}


def format_results(results: list, query_name: str) -> str:
    """Format query results for display"""
    if not results:
        return "No results found."
    
    output = []
    output.append(f"\nQuery: {query_name}")
    output.append(f"Results: {len(results)} record(s)")
    output.append("-" * 80)
    
    # Get column names from first result
    if results:
        columns = list(results[0].keys())
        output.append(" | ".join(columns))
        output.append("-" * 80)
        
        # Display first 10 results
        for i, record in enumerate(results[:10]):
            row = []
            for col in columns:
                value = record.get(col, "N/A")
                # Format lists and complex types
                if isinstance(value, list):
                    value = ", ".join(str(v) for v in value)
                row.append(str(value))
            output.append(" | ".join(row))
        
        if len(results) > 10:
            output.append(f"... and {len(results) - 10} more results")
    
    return "\n".join(output)


def process_query(classifier: IntentClassifier, retrieval: GraphRetrieval, 
                  query: str, verbose: bool = True) -> dict:
    """
    Process a user query: classify intent, retrieve from graph, return results.
    
    Args:
        classifier: IntentClassifier instance
        retrieval: GraphRetrieval instance
        query: User query string
        verbose: Whether to print detailed information
        
    Returns:
        Dictionary with intent, entities, query_name, and results
    """
    if verbose:
        print(f"\n{'='*80}")
        print(f"Processing Query: {query}")
        print(f"{'='*80}")
    
    # Step 1: Classify intent and extract entities
    intent, metadata = classifier.classify(query)
    entities = metadata.get("entities", {})
    
    if verbose:
        print(f"Intent: {intent.value}")
        print(f"Entities: {entities}")
    
    # Step 2: Select appropriate query template
    # Convert intent enum value to uppercase to match query templates
    intent_str = intent.value.upper()
    query_name = retrieval.select_query(intent_str, entities)
    
    if not query_name:
        if verbose:
            print("No suitable query template found for this intent and entities.")
        return {
            "intent": intent.value,
            "entities": entities,
            "query_name": None,
            "results": []
        }
    
    if verbose:
        query_info = retrieval.get_query_info(query_name)
        print(f"Selected Query: {query_name}")
        print(f"Description: {query_info.get('description', 'N/A')}")
        
        # Generate and display the actual Cypher query
        query_name_gen, cypher_query, params = retrieval.generate_cypher_query(intent_str, entities)
        if cypher_query:
            print(f"\nGenerated Cypher Query:")
            print("-" * 80)
            print(cypher_query)
            print("-" * 80)
            print(f"Parameters: {params}")
    
    # Step 3: Parameterize and execute query
    try:
        results = retrieval.retrieve(intent_str, entities)
        
        if verbose:
            print(format_results(results, query_name))
        
        return {
            "intent": intent.value,
            "entities": entities,
            "query_name": query_name,
            "results": results
        }
    
    except Exception as e:
        if verbose:
            print(f"Error executing query: {e}")
        return {
            "intent": intent.value,
            "entities": entities,
            "query_name": query_name,
            "results": [],
            "error": str(e)
        }


def main():
    """Main function demonstrating the retrieval pipeline"""
    
    # Load configuration
    config = load_config()
    
    if not config:
        print("Could not load configuration. Please check config.txt file.")
        print("\nExample queries that would be processed:")
        example_queries = [
            "How many points did Mohamed Salah score in 2022-23?",
            "Who are the top 10 defenders in the 2022-23 season?",
            "Show me Arsenal fixtures for gameweek 10 in 2022-23",
            "What games are in gameweek 5 of 2022-23?",
            "Which teams played in the 2022-23 season?",
            "Compare Mohamed Salah vs Erling Haaland in 2022-23",
            "Find players who play as defender",
            "How many total gameweeks are in 2022-23?",
            "Who scored the most goals in 2022-23?",
            "Show me all players"
        ]
        
        classifier = IntentClassifier()
        retrieval = GraphRetrieval.__new__(GraphRetrieval)
        retrieval.query_templates = retrieval._initialize_query_templates()
        
        for query in example_queries:
            print(f"\nQuery: {query}")
            intent, metadata = classifier.classify(query)
            intent_str = intent.value.upper()
            query_name = retrieval.select_query(intent_str, metadata["entities"])
            print(f"  Intent: {intent.value} -> {intent_str}")
            print(f"  Entities: {metadata['entities']}")
            print(f"  Selected Query: {query_name}")
        
        return
    
    # Initialize components
    try:
        classifier = IntentClassifier()
        retrieval = GraphRetrieval(
            uri=config.get("URI", ""),
            username=config.get("USERNAME", ""),
            password=config.get("PASSWORD", ""),
            database=config.get("DATABASE", "neo4j")
        )
        
        print("Graph Retrieval Layer - Baseline Retrieval")
        print("=" * 80)
        print("Successfully connected to Neo4j database")
        
    except Exception as e:
        print(f"Error initializing components: {e}")
        return
    
    # Example queries covering different intents
    test_queries = [
        # Player Performance
        "How many points did Mohamed Salah score in 2022-23?",
        "Show me Erling Haaland's stats for gameweek 10 in 2022-23",
        
        # Player Recommendations
        "Who are the top 10 defenders in the 2022-23 season?",
        "Get me the best forwards for 2022-23",
        "Top 5 players overall in 2022-23",
        
        # Fixture Queries
        "Show me Arsenal fixtures for gameweek 10 in 2022-23",
        "What fixtures does Liverpool have in 2022-23?",
        
        # Gameweek Queries
        "What games are in gameweek 5 of 2022-23?",
        "Show me all fixtures in gameweek 1 for 2022-23",
        
        # Season Queries
        "Which teams played in the 2022-23 season?",
        
        # Comparison Queries
        "Compare Mohamed Salah vs Erling Haaland in 2022-23",
        
        # Position Queries
        "Find players who play as defender",
        "Show me all goalkeepers in 2022-23",
        
        # Statistics Queries
        "How many total gameweeks are in 2022-23?",
        "Who scored the most goals in 2022-23?",
        "Who has the most assists in 2022-23?",
        
        # Entity Search
        "Search for players named Salah",
        "Show me all players"
    ]
    
    # Process each query
    for query in test_queries:
        try:
            process_query(classifier, retrieval, query, verbose=True)
        except Exception as e:
            print(f"Error processing query '{query}': {e}")
    
    # Close connection
    retrieval.close()
    print("\n" + "="*80)
    print("Retrieval session completed.")


if __name__ == "__main__":
    main()

