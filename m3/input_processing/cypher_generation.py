"""
Test script to show Cypher query generation with entities
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import time
from dotenv import load_dotenv
from openai import OpenAI
from input_processing.intent_classification import IntentClassifier
from graph_retrieval_layer.graph_retrieval import GraphRetrieval
from typing import Dict, List, Any, Optional
from langchain_core.language_models.llms import LLM  # you already have this import
import os
from huggingface_hub import InferenceClient
from pydantic import Field
from embeddings.test_embedding import embed_query
from google import genai



load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


openai_client = OpenAI(api_key=OPENAI_API_KEY)
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

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

# ---- OpenAI helpers (no LangChain wrapper needed) ----

OPENAI_PRICES_USD = {
    # adjust to whatever models you use and current prices
    "gpt-3.5-turbo": {"input": 0.0005 / 1000, "output": 0.0015 / 1000},
    "gpt-4o-mini":   {"input": 0.00015 / 1000, "output": 0.0006 / 1000},
    "gpt-4o":        {"input": 0.005 / 1000, "output": 0.015 / 1000},
}

def call_openai_model(model: str, prompt: str) -> Dict[str, Any]:
    """
    Call an OpenAI chat model and return text + usage + timing.
    """
    start = time.perf_counter()
    resp = openai_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=500,
    )
    elapsed = time.perf_counter() - start

    msg = resp.choices[0].message.content
    usage = getattr(resp, "usage", None)

    prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None
    completion_tokens = getattr(usage, "completion_tokens", None) if usage else None
    total_tokens = getattr(usage, "total_tokens", None) if usage else None

    cost_usd = None
    if total_tokens is not None and model in OPENAI_PRICES_USD:
        prices = OPENAI_PRICES_USD[model]
        # rough split of input/output if we don't have them separately
        if prompt_tokens is None or completion_tokens is None:
            prompt_tokens = int(total_tokens * 0.6)
            completion_tokens = total_tokens - prompt_tokens
        cost_usd = (
            prompt_tokens * prices["input"]
            + completion_tokens * prices["output"]
        )

    return {
        "text": msg,
        "response_time": elapsed,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cost_usd": cost_usd,
    }

def _merge_and_dedup_rows(
    baseline_rows: List[Dict[str, Any]],
    emb_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Merge baseline + embedding rows and remove duplicates.

    For FPL we try to de-duplicate by (name, season, GW) when present.
    Otherwise we fall back to stringifying the dict.
    """
    merged: List[Dict[str, Any]] = []
    seen_keys = set()

    def make_key(row: Dict[str, Any]):
        if all(k in row for k in ("name", "season", "GW")):
            return ("triple", str(row["name"]), str(row["season"]), str(row["GW"]))
        return ("raw", repr(sorted(row.items())))

    for r in baseline_rows + emb_rows:
        key = make_key(r)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        merged.append(r)
    return merged


def run_models_for_query(
    user_query: str,
    models_to_run: List[str],
    retrieval_method: str = "baseline",           # "baseline" | "embeddings" | "hybrid"
    emb_model_key: Optional[str] = None,          # "minilm" | "mpnet"
    emb_indexes: Optional[Dict[str, Any]] = None, # FAISS indexes dict from Streamlit
) -> Dict[str, Any]:
    """
    1. Classify intent & entities for user_query
    2. Depending on retrieval_method:
         - baseline: run Cypher over Neo4j only
         - embeddings: run semantic search over FAISS only
         - hybrid: run both, then merge + de-dup rows
    3. Build table_str from the *combined* context
    4. Run all requested LLMs on (user_query + table_str)
    5. Return everything in one dict for Streamlit
    """

    retrieval_method = retrieval_method.lower().strip()
    use_baseline = retrieval_method in ("baseline", "hybrid")
    use_embeddings = retrieval_method in ("embeddings", "hybrid")

    # Basic validation
    # if use_embeddings and (emb_indexes is None or not emb_indexes):
    #     return {
    #         "user_query": user_query,
    #         "intent": "unknown",
    #         "entities": {},
    #         "error": "Embeddings retrieval requested but FAISS indexes are not initialized.",
    #         "retrieval_method": retrieval_method,
    #     }
    # if use_embeddings and not emb_model_key:
    #     return {
    #         "user_query": user_query,
    #         "intent": "unknown",
    #         "entities": {},
    #         "error": "Embeddings retrieval requested but no embedding model was selected.",
    #         "retrieval_method": retrieval_method,
    #     }

    # --- Intent classification (we always do this) ---
    classifier = IntentClassifier()
    intent, metadata = classifier.classify(user_query)
    entities = metadata.get("entities", {})
    intent_str = intent.value.upper()

    # Try to extract season / gameweek for optional embedding filtering
    season_entity: Optional[str] = None
    gw_entity: Optional[str] = None
    for key in ("season", "seasons"):
        if key in entities and entities[key]:
            season_entity = str(entities[key][0])
            break
    for key in ("gameweek", "gw", "gameweeks"):
        if key in entities and entities[key]:
            gw_entity = str(entities[key][0])
            break

    # --- Baseline retrieval ---
    baseline_results: List[Dict[str, Any]] = []
    query_name: Optional[str] = None
    cypher_query: Optional[str] = None
    clean_params: Dict[str, Any] = {}

    retrieval = None
    if use_baseline:
        uri = "neo4j+s://6fe2fa9b.databases.neo4j.io"
        username = "neo4j"
        password = "6VR8BRVu3AJPCP8QZio4ifSrdYoHb1eHPDGcVBmD0kc"
        database = "neo4j"
        retrieval = GraphRetrieval(uri, username, password, database)

    try:
        if use_baseline and retrieval is not None:
            query_name, cypher_query, params = retrieval.generate_cypher_query(
                intent_str, entities
            )

            if query_name:
                query_template, exec_params = retrieval.parameterize_query(
                    query_name, entities
                )
                clean_params = {
                    k: (None if v is None else v) for k, v in exec_params.items()
                }
                baseline_results = retrieval.execute_query(query_template, clean_params)
            else:
                # If baseline chosen but nothing matched, keep results empty
                pass

        # --- Embedding retrieval (per-row player-GW docs) ---
        embedding_rows: List[Dict[str, Any]] = []
        if use_embeddings:
            if emb_model_key == "minilm":
                emb_model_name = "all-MiniLM-L6-v2"
                emb_embedding_dim = 384
            elif emb_model_key == "mpnet":
                emb_model_name = "all-mpnet-base-v2"
                emb_embedding_dim = 768
            else:
                raise ValueError(f"Invalid embedding model key: {emb_model_key}")
            hits = embed_query(query=user_query, model_name=emb_model_name, embedding_dim=emb_embedding_dim, top_k=10)
            for result in hits:
                meta = result['metadata']
                embedding_rows.append(dict(meta))

        # --- Merge context depending on method ---
        if retrieval_method == "baseline":
            combined_rows = list(baseline_results)
        elif retrieval_method == "embeddings":
            combined_rows = list(embedding_rows)
        else:  # hybrid
            combined_rows = _merge_and_dedup_rows(baseline_results, embedding_rows)

        table_str = (
            format_results_table(combined_rows) if combined_rows else "No results found."
        )

        # --- Run selected LLMs on the combined context ---
        models_output: Dict[str, Any] = {}

        if "Gemma" in models_to_run:
            start = time.perf_counter()
            try:
                prompt = build_llm_prompt(user_query, table_str)
                gemma_text = gemma_llm.invoke(prompt)
                elapsed = time.perf_counter() - start
                models_output["Gemma (google/gemma-2-2b-it)"] = {
                    "answer": gemma_text,
                    "response_time": elapsed,
                    "prompt_tokens": None,
                    "completion_tokens": None,
                    "total_tokens": None,
                    "cost_usd": None,
                }
            except Exception as e:
                models_output["Gemma (google/gemma-2-2b-it)"] = {"error": str(e)}

        if "GPT-3.5" in models_to_run:
            try:
                prompt = build_llm_prompt(user_query, table_str)
                info = call_openai_model("gpt-3.5-turbo", prompt)
                models_output["GPT-3.5 (gpt-3.5-turbo)"] = {
                    "answer": info["text"],
                    "response_time": info["response_time"],
                    "prompt_tokens": info["prompt_tokens"],
                    "completion_tokens": info["completion_tokens"],
                    "total_tokens": info["total_tokens"],
                    "cost_usd": info["cost_usd"],
                }
            except Exception as e:
                models_output["GPT-3.5 (gpt-3.5-turbo)"] = {"error": str(e)}

        if "GPT-4" in models_to_run:
            try:
                prompt = build_llm_prompt(user_query, table_str)
                info = call_openai_model("gpt-4o", prompt)
                models_output["GPT-4 (gpt-4o)"] = {
                    "answer": info["text"],
                    "response_time": info["response_time"],
                    "prompt_tokens": info["prompt_tokens"],
                    "completion_tokens": info["completion_tokens"],
                    "total_tokens": info["total_tokens"],
                    "cost_usd": info["cost_usd"],
                }
            except Exception as e:
                models_output["GPT-4 (gpt-4o)"] = {"error": str(e)}
        
        if "Gemini" in models_to_run:
            try:
                prompt = build_llm_prompt(user_query, table_str)

                import time
                start = time.time()

                info = gemini_client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                )

                end = time.time()
                response_time = round(end - start, 4)

                models_output["Gemini (gemini-2.5-flash)"] = {
                    "answer": info.text,
                    "response_time": response_time,
                    "prompt_tokens": None,
                    "completion_tokens": None,
                    "total_tokens": None,
                    "cost_usd": None,
                }

            except Exception as e:
                models_output["Gemini (gemini-2.5-flash)"] = {"error": str(e)}

        
        return {
            "user_query": user_query,
            "intent": intent.value,
            "entities": entities,
            "retrieval_method": retrieval_method,
            "query_name": query_name,
            "cypher_query": cypher_query,
            "exec_params": clean_params if use_baseline else {},
            "baseline_results": baseline_results,
            "embedding_results": embedding_rows,
            "raw_results": combined_rows,
            "table_str": table_str,
            "models": models_output,
        }

    finally:
        if retrieval is not None:
            retrieval.close()


def build_llm_prompt(user_query: str, table_str: str) -> str:
    return f"""
You are a Fantasy Premier League (FPL) assistant.

The user asked this question:
\"\"\"{user_query}\"\"\"

You have the following context table, coming from the FPL knowledge graph
(baseline Cypher queries) and/or semantic embedding search over per-player
gameweek rows:

{table_str}

Instructions:
- Use ONLY the information in the table to answer.
- Answer clearly in good English.
- If the table is empty or doesn't contain enough information, say that you
  don't have the data to answer exactly.
- Be concise but informative, and explain the key numbers in a friendly way.

Now write your answer to the user:
"""


def explain_results_with_llm(user_query: str, table_str: str, llm: LLM) -> str:
    prompt = build_llm_prompt(user_query, table_str)
    return llm.invoke(prompt)



def test_cypher_generation2(Queries: List[str]):
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



def test_cypher_generation3(Queries: List[str]):
    """Test Cypher query generation for various queries"""
    
    classifier = IntentClassifier()
    
    # Create mock retrieval instance (without connection)
    retrieval = GraphRetrieval.__new__(GraphRetrieval)
    retrieval.query_templates = retrieval._initialize_query_templates()
    
   
    
    for query in Queries:
        # Classify intent and extract entities
        intent, metadata = classifier.classify(query)
        entities = metadata["entities"]
        # Generate Cypher query
        intent_str = intent.value.upper()
        query_name, cypher_query, params = retrieval.generate_cypher_query(intent_str, entities)
        
        if query_name:
            query_info = retrieval.get_query_info(query_name)
            
        else:
            return "Expected no template for unknown query {query!r}, got {template!r}"


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



def test_cypher_generation(Queries: List[str]) -> List[Dict[str, Any]]:
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
        "List all the player in arsenal in gameweek 5 in 2021-22",
        "Top forwards ",

    ]
    test_queries_5 = [
        'What is the average points per game for all players?',
        'Show me overall stats for the 2022-23 season',
        'Which season had the most total goals?'
    ]
    test_queries_6 = [ 
        "How many total gameweeks are there?",
        "What is the highest points scored by a player?",
        "Which season had the most total goals?",
    ]
    # test_cypher_generation(test_queries)
    # test_cypher_generation2(test_queries+test_queries_2+test_queries_3+test_queries_4+test_queries_5+test_queries_6)
    test_cypher_generation2(test_queries_6)
