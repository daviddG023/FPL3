"""
Test script to show Cypher query generation with entities
"""

from dotenv import load_dotenv
from intent_classification_chat import IntentClassifier
from graph_retrieval import GraphRetrieval
from typing import Dict, List, Any

from typing import List, Dict, Any
from langchain_core.language_models.llms import LLM  # you already have this import

from huggingface_hub import InferenceClient
from langchain_core.language_models.llms import LLM
from pydantic import Field
from typing import Any, Optional, List

load_dotenv()
import os
HF_TOKEN = os.getenv("HF_TOKEN")

# Initialize Hugging Face client for Gemma
client = InferenceClient(
    model="google/gemma-2-2b-it",
    token=HF_TOKEN,
)

class GemmaWrapper(LLM):
    """Wrapper that lets LangChain call Gemma via HuggingFace Inference API."""
    client: Any = Field(...)
    max_tokens: int = 500

    @property
    def _llm_type(self) -> str:
        return "gemma_hf_api"

    def _call(self, prompt: str, stop: Optional[List[str]] = None):
        response = self.client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=0.2,
        )
        return response.choices[0].message["content"]

# ✅ This is what your test_cypher_generation2() will use
gemma_llm = GemmaWrapper(client=client)


def explain_results_with_llm(user_query: str, table_str: str, llm: LLM) -> str:
    """
    Use the LLM (Gemma via GemmaWrapper) to answer the user's query
    based on the query results table.
    """
    prompt = f"""
You are a Fantasy Premier League (FPL) assistant.

The user asked this question:
\"\"\"{user_query}\"\"\"

You have the following table of data that comes from Neo4j:

{table_str}

Instructions:
- Use ONLY the information in the table to answer.
- Answer clearly in good English.
- If the table is empty or doesn't contain enough information, say that you
  don't have the data to answer exactly.
- Be concise but informative, and explain the key numbers in a friendly way.

Now write your answer to the user:
"""
    # ✅ Use LangChain's invoke API instead of calling the object directly
    answer = llm.invoke(prompt)
    return answer


def test_cypher_generation(Queries: List[str]):
    """Test Cypher query generation for various queries"""
    
    classifier = IntentClassifier()
    
    # Create mock retrieval instance (without connection)
    retrieval = GraphRetrieval.__new__(GraphRetrieval)
    retrieval.query_templates = retrieval._initialize_query_templates()
    
    print("=" * 100)
    print("Cypher Query Generation Test")
    print("=" * 100)
    
    for query in Queries:
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




def format_results_table(results: List[Dict[str, Any]]) -> str:
    """
    Format query results as a table.
    
    Args:
        results: List of result dictionaries
        
    Returns:
        Formatted table string
    """
    if not results:
        return "No results found."
    
    # Get column names from first result
    columns = list(results[0].keys())
    
    # Calculate column widths
    col_widths = {}
    for col in columns:
        # Start with column name width
        col_widths[col] = len(str(col))
        # Check all values in this column
        for row in results:
            value = row.get(col, "")
            # Format lists and complex types
            if isinstance(value, list):
                value_str = ", ".join(str(v) for v in value)
            elif value is None:
                value_str = "NULL"
            else:
                value_str = str(value)
            col_widths[col] = max(col_widths[col], len(value_str))
        # Add padding
        col_widths[col] = min(col_widths[col] + 2, 50)  # Max width 50
    
    # Build table
    lines = []
    
    # Header
    header = " | ".join(str(col).ljust(col_widths[col]) for col in columns)
    lines.append(header)
    lines.append("-" * len(header))
    
    # Rows
    for row in results:
        row_values = []
        for col in columns:
            value = row.get(col, "")
            # Format lists and complex types
            if isinstance(value, list):
                value_str = ", ".join(str(v) for v in value)
            elif value is None:
                value_str = "NULL"
            else:
                value_str = str(value)
            # Truncate if too long
            if len(value_str) > col_widths[col]:
                value_str = value_str[:col_widths[col]-3] + "..."
            row_values.append(value_str.ljust(col_widths[col]))
        lines.append(" | ".join(row_values))
    
    return "\n".join(lines)


def load_config(path="config.txt"):
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


from typing import List, Dict, Any

def test_cypher_generation2(Queries: List[str]) -> List[Dict[str, Any]]:
    """
    Test Cypher query generation and execution for various queries.
    
    After executing each query, this function:
      - formats the results as a table string
      - sends (user_query + table) to the LLM (Gemma) to get a natural-language answer
      - returns a list of dicts with everything useful
    """
    
    # Load config and initialize retrieval with actual connection
    uri = "neo4j+s://6fe2fa9b.databases.neo4j.io"
    username = "neo4j"
    password = "6VR8BRVu3AJPCP8QZio4ifSrdYoHb1eHPDGcVBmD0kc"
    database = "neo4j"
    retrieval = GraphRetrieval(uri, username, password, database)
    classifier = IntentClassifier()
    
    results_summary: List[Dict[str, Any]] = []
    
    print("=" * 100)
    print("Cypher Query Generation and Execution Test")
    print("=" * 100)
    
    for query in Queries:
        print(f"\n{'='*100}")
        print(f"Query: {query}")
        print(f"{'='*100}")
        
        try:
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

                # Execute query and get results
                print(f"\n{'='*100}")
                print("Executing Query...")
                print(f"{'='*100}")
                
                try:
                    # Get the actual query template and parameters for execution
                    query_template, exec_params = retrieval.parameterize_query(query_name, entities)
                    
                    # Clean up None values in parameters (replace with actual None for Neo4j)
                    clean_params = {k: (None if v is None else v) for k, v in exec_params.items()}
                    
                    # Execute the query
                    results = retrieval.execute_query(query_template, clean_params)
                    
                    print(f"\n✅ Query executed successfully!")
                    print(f"📊 Results: {len(results)} record(s) found\n")
                    
                    # Format results as text table
                    if results:
                        table_str = format_results_table(results)
                        print(table_str)
                    else:
                        table_str = "No results returned."
                        print(table_str)

                    # 🔥 Call LLM to explain the results in good English
                    try:
                        llm_answer = explain_results_with_llm(
                            user_query=query,
                            table_str=table_str,
                            llm=gemma_llm,     # GemmaWrapper instance from your RAG code
                        )
                        print("\n🧠 LLM Answer:")
                        print(llm_answer)
                    except Exception as llm_err:
                        llm_answer = f"Error calling LLM: {llm_err}"
                        print(f"\n❌ {llm_answer}")

                    # Store everything for external use (e.g., Streamlit)
                    results_summary.append({
                        "user_query": query,
                        "intent": intent.value,
                        "entities": entities,
                        "query_name": query_name,
                        "cypher_query": cypher_query,
                        "exec_query_template": query_template,
                        "exec_params": clean_params,
                        "raw_results": results,
                        "table_str": table_str,
                        "llm_answer": llm_answer,
                    })
                        
                except Exception as exec_error:
                    print(f"\n❌ Error executing query: {exec_error}")
                    print(f"\nQuery Template:")
                    print(query_template[:500] + "..." if len(query_template) > 500 else query_template)
                    print(f"\nParameters: {exec_params}")

                    results_summary.append({
                        "user_query": query,
                        "intent": intent.value,
                        "entities": entities,
                        "query_name": query_name,
                        "error": str(exec_error),
                    })
                    
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
                
                results_summary.append({
                    "user_query": query,
                    "intent": intent.value,
                    "entities": entities,
                    "query_name": None,
                    "error": "No suitable query template found",
                })
        
        except Exception as e:
            print(f"\n❌ Error processing query: {e}")
            import traceback
            traceback.print_exc()
            
            results_summary.append({
                "user_query": query,
                "error": str(e),
            })
    
    # Close connection
    retrieval.close()
    print(f"\n{'='*100}")
    print("Test completed. Database connection closed.")
    print(f"{'='*100}")
    
    # ✅ Return structured data for further use
    return results_summary


if __name__ == "__main__":
    test_queries = [
        "How many points did Mohamed Salah score in 2022-23?",
        "Who are the top 10 defenders in the 2022-23 season?",
        "Show me arsenal fixtures for gameweek 10 in 2022-23",
        "What games are in gameweek 5 of 2022-23?",
        "Compare Mohamed Salah vs Erling Haaland in 2022-23 in gameweek 10",
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
    test_queries_3 = [
        "List all teams",
        "List all positions",
        "List all seasons",
        "List all gameweeks",
        "List all fixtures",
        "List all teams in 2022-23",
        "List all fixtures in gameweek 5 in 2022-23",
    ]
    test_queries_4 = [
        "List all the player in arsenal",
        "Top forwards in 2022-23",
    ]
    
    # test_cypher_generation(test_queries)
    test_cypher_generation2(test_queries)
    # test_cypher_generation2()
