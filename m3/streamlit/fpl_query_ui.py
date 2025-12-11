"""
FPL Query Interface - Streamlit UI
Part 4: Build a UI for the FPL Knowledge Graph Query System

This Streamlit application provides a user-friendly interface for querying
the FPL Knowledge Graph using natural language queries.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]   # this is the m3 folder
sys.path.append(str(ROOT))


from embeddings.fpl_feature_embeddings import init_player_gw_indexes
import streamlit as st
import sys
import os
from pathlib import Path
from input_processing.cypher_generation import test_cypher_generation, run_models_for_query
# Add the Input Preprocessing directory to the path
sys.path.append(str(Path(__file__).parent / "Input Preprocessing"))

from input_processing.intent_classification import IntentClassifier
from graph_retrieval_layer.graph_retrieval import GraphRetrieval
import pandas as pd

def load_config():
    """Load Neo4j configuration from config file"""
    config = {}
    # Try multiple possible paths
    possible_paths = [
        "m2/config.txt",  # From project root
        "../m2/config.txt",  # From m3 directory
        Path(__file__).parent.parent / "m2" / "config.txt"  # Absolute path
    ]
    
    config_path = None
    for path in possible_paths:
        if isinstance(path, Path):
            if path.exists():
                config_path = path
                break
        else:
            if os.path.exists(path):
                config_path = path
                break
    
    if not config_path:
        return {}
    
    try:
        with open(config_path, "r") as f:
            for line in f:
                line = line.strip()
                if "=" in line:
                    key, value = line.split("=", 1)
                    config[key] = value.strip().replace('"', '')
        return config
    except Exception as e:
        if hasattr(st, 'error'):
            st.error(f"Error loading config: {e}")
        return {}


def format_results_as_dataframe(results: list) -> pd.DataFrame:
    """Convert query results to a pandas DataFrame for better display"""
    if not results:
        return pd.DataFrame()
    
    return pd.DataFrame(results)


def format_results_text(results: list, query_name: str) -> str:
    """Format query results as text"""
    if not results:
        return "No results found."
    
    output = []
    output.append(f"**Query:** {query_name}")
    output.append(f"**Results:** {len(results)} record(s)\n")
    
    if results:
        columns = list(results[0].keys())
        output.append(" | ".join(columns))
        output.append(" | ".join(["---"] * len(columns)))
        
        for record in results[:20]:  # Limit to 20 for display
            row = []
            for col in columns:
                value = record.get(col, "N/A")
                if isinstance(value, list):
                    value = ", ".join(str(v) for v in value)
                row.append(str(value))
            output.append(" | ".join(row))
        
        if len(results) > 20:
            output.append(f"\n... and {len(results) - 20} more results")
    
    return "\n".join(output)


def initialize_session_state():
    """Initialize session state variables"""
    if 'classifier' not in st.session_state:
        st.session_state.classifier = IntentClassifier()
    
    # if "emb_indexes" not in st.session_state:
    #     st.session_state.emb_indexes = init_player_gw_indexes(
    #         csv_path="fpl_two_seasons.csv",
    #         faiss_dir="faiss_indexes_player_gw",
    #     )
    
    if 'retrieval' not in st.session_state:
        config = load_config()
        if config:
            try:
                st.session_state.retrieval = GraphRetrieval(
                    uri=config.get("URI", ""),
                    username=config.get("USERNAME", ""),
                    password=config.get("PASSWORD", ""),
                    database=config.get("DATABASE", "neo4j")
                )
                st.session_state.db_connected = True
            except Exception as e:
                st.session_state.db_connected = False
                st.session_state.db_error = str(e)
        else:
            st.session_state.db_connected = False
    
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []


def process_query(query: str) -> dict:
    """Process a user query and return results"""
    if not st.session_state.db_connected:
        return {
            "intent": "UNKNOWN",
            "entities": {},
            "query_name": None,
            "results": [],
            "error": "Database not connected. Please check configuration."
        }
    
    # Classify intent
    intent, metadata = st.session_state.classifier.classify(query)
    entities = metadata.get("entities", {})
    
    # Select and execute query
    intent_str = intent.value.upper()
    query_name = st.session_state.retrieval.select_query(intent_str, entities)
    
    if not query_name:
        return {
            "intent": intent.value,
            "entities": entities,
            "query_name": None,
            "results": [],
            "error": "No suitable query template found for this intent and entities."
        }
    
    try:
        results = st.session_state.retrieval.retrieve(intent_str, entities)
        
        # Generate Cypher query for display
        query_name_gen, cypher_query, params = st.session_state.retrieval.generate_cypher_query(
            intent_str, entities
        )
        
        return {
            "intent": intent.value,
            "entities": entities,
            "query_name": query_name,
            "cypher_query": cypher_query,
            "params": params,
            "results": results
        }
    except Exception as e:
        return {
            "intent": intent.value,
            "entities": entities,
            "query_name": query_name,
            "results": [],
            "error": str(e)
        }
    

def main():
    st.set_page_config(
        page_title="FPL Query Interface",
        page_icon="⚽",
        layout="wide",
        initial_sidebar_state="expanded",
    )

     # Custom CSS for FPL theme (green colors)
    st.markdown("""
        <style>
        .main-header {
            background: linear-gradient(90deg, #37003c 0%, #00ff87 100%);
            padding: 1rem;
            border-radius: 10px;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
        }
        .stButton>button {
            background-color: #00ff87;
            color: #37003c;
            font-weight: bold;
            border-radius: 5px;
            border: none;
        }
        .stButton>button:hover {
            background-color: #00cc6a;
            color: white;
        }
        .query-box {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 5px;
            border-left: 4px solid #00ff87;
        }
        .result-box {
            background-color: #ffffff;
            padding: 1rem;
            border-radius: 5px;
            border: 1px solid #e0e0e0;
            margin-top: 1rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # (You can still keep this if you like the status / history logic)
    initialize_session_state()
    

    st.markdown(
        """
        <div class="main-header">
            <h1>⚽ FPL Knowledge Graph Query Interface</h1>
            <p>Ask questions about Fantasy Premier League players, teams, fixtures, and statistics</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ───────────────── Sidebar ─────────────────
    with st.sidebar:
        st.header("📊 About")
        st.markdown(
            """
            This interface allows you to query the FPL Knowledge Graph using natural language.
            """
        )

        st.header("🔧 Status")
        if st.session_state.get("db_connected", False):
            st.success("✅ Database Connected")
        else:
            st.error("❌ Database Not Connected")
            if st.session_state.get("db_error"):
                st.error(f"Error: {st.session_state.db_error}")

        # === NEW: Retrieval method selection ===
        st.header("🔍 Retrieval Method")
        retrieval_choice = st.radio(
            "Choose retrieval approach",
            [
                "Baseline (Cypher only)",
                "Embeddings only",
                "Hybrid (Baseline + Embeddings)",
            ],
            index=0,
        )

        # Map human label -> internal code
        if retrieval_choice.startswith("Baseline"):
            retrieval_method = "baseline"
        elif retrieval_choice.startswith("Embeddings"):
            retrieval_method = "embeddings"
        else:
            retrieval_method = "hybrid"

        # === NEW: Embedding model selection (only when embeddings are used) ===
        emb_model_key = None
        if retrieval_method in ("embeddings", "hybrid"):
            st.subheader("🧮 Embedding Model")
            emb_label = st.selectbox(
                "Select embedding model",
                [
                    "MiniLM (all-MiniLM-L6-v2) – fast",
                    "MPNet (all-mpnet-base-v2) – more accurate",
                ],
            )
            emb_model_key = "minilm" if emb_label.startswith("MiniLM") else "mpnet"

        st.header("🤖 LLM Model Selection")
        model_choices = ["Gemma", "GPT-3.5", "GPT-4", "Gemini"]
        selected_models = st.multiselect(
            "Choose LLMs to run",
            model_choices,
            default=["Gemma", "GPT-3.5", "GPT-4", "Gemini"],
        )

        st.header("📝 Query History")
        if st.session_state.query_history:
            for i, hist_query in enumerate(
                reversed(st.session_state.query_history[-5:])
            ):
                if st.button(
                    f"Query {len(st.session_state.query_history) - i}: {hist_query[:30]}...",
                    key=f"hist_{i}",
                ):
                    st.session_state.current_query = hist_query
        else:
            st.info("No queries yet")

    # ───────────────── Main layout ─────────────────
    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader("💬 Enter Your Query")
        query = st.text_input(
            "Ask a question about FPL:",
            value=st.session_state.get("current_query", ""),
            placeholder="e.g., How many points did Mohamed Salah score in 2022-23?",
            key="query_input",
        )

        c1, c2 = st.columns([1, 5])
        with c1:
            submit_button = st.button(
                "🔍 Query", type="primary", use_container_width=True
            )
        with c2:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.current_query = ""
                st.rerun()

    with col2:
        st.subheader("💡 Quick Examples")
        example_queries = [ "Top 10 players in 2022-23", "Arsenal fixtures in gameweek 10", "Compare Mohamed Salah vs Erling Haaland", "Best defenders in 2022-23", "Games in gameweek 5", "Who are the top 10 defenders in the 2022-23 season?", "Show me arsenal fixtures for gameweek 10 in 2022-23", "What games are in gameweek 5 of 2022-23?", "Compare Mohamed Salah vs Erling Haaland in 2022-23 in gameweek 10", "Show me Erling Haaland's stats for gameweek 10 in 2022-23", "How many points did Mohamed Salah score in 2022-23?", "Who are the best defenders to pick in GW5?", "Show me all fixtures for Arsenal", "What games are in gameweek 5?", "Which teams played in the 2022-23 season?", "Compare Mohamed Salah vs Erling Haaland this season", "Find players who play as defender", "How many total gameweeks are there?", "What is the highest points scored by a player?", "Show me all players", "List all teams", "List all positions", "List all seasons", "List all gameweeks", "List all fixtures", "List all teams in 2022-23", "List all fixtures in gameweek 5 in 2022-23", "List all the player in arsenal", "Top forwards in 2022-23", ]
        selected_example = st.selectbox(
            "Choose an example query:",
            ["Select an example"] + example_queries,   # Empty first option
            index=0
        )

        
        if "example_ran" not in st.session_state:
            st.session_state.example_ran = False

        if selected_example != "Select an example":
            st.session_state.current_query = selected_example
            if not st.session_state.example_ran:
                st.session_state.example_ran = True
                selected_example = "Select an example"
                st.rerun()


    # ───────────────── Handle submit ─────────────────
    if submit_button and query:
        if not selected_models:
            st.warning("Please select at least one LLM model.")
        elif retrieval_method in ("embeddings", "hybrid") and not emb_model_key:
            st.warning("Please select an embedding model.")
        else:
            with st.spinner("Processing your query..."):
                try:
                    result = run_models_for_query(
                        user_query=query,
                        models_to_run=selected_models,
                        retrieval_method=retrieval_method,      # NEW
                        emb_model_key=emb_model_key,            # NEW
                        emb_indexes=st.session_state.emb_indexes,  # NEW
                    )
                except Exception as e:
                    result = {"error": f"Error running pipeline: {e}"}

            if query not in st.session_state.query_history:
                st.session_state.query_history.append(query)

            # ── Top-level info ──
            st.markdown("---")
            st.subheader("📋 Query Analysis")

            col_intent, col_entities = st.columns(2)
            with col_intent:
                st.markdown("**Detected Intent:**")
                intent_display = result.get("intent", "unknown")
                st.info(f"**{str(intent_display).replace('_', ' ').title()}**")

                rm = result.get("retrieval_method", retrieval_method)
                st.markdown(f"**Retrieval used:** `{rm}`")

            with col_entities:
                st.markdown("**Extracted Entities:**")
                entities = result.get("entities", {})
                if entities:
                    lines = []
                    for key, values in entities.items():
                        if not values:
                            continue
                        if isinstance(values, (list, tuple)):
                            v_str = ", ".join(str(v) for v in values)
                        else:
                            v_str = str(values)
                        lines.append(f"**{key.title()}:** {v_str}")
                    if lines:
                        st.info("\n".join(lines))
                    else:
                        st.info("No entities extracted")
                else:
                    st.info("No entities extracted")

            # ── Cypher query (only if baseline used) ──
            if result.get("query_name") and result.get("cypher_query"):
                st.markdown("---")
                st.subheader("🔧 Baseline Cypher Details")
                with st.expander("View Cypher Query"):
                    st.code(result["cypher_query"], language="cypher")
                    if result.get("exec_params"):
                        st.json(result["exec_params"])

            # ── Results + LLM answers ──
            st.markdown("---")
            st.subheader("📊 Retrieved Context & LLM Comparison")

            if result.get("error"):
                st.error(f"❌ Error: {result['error']}")
            else:
                baseline_rows = result.get("baseline_results") or []
                emb_rows = result.get("embedding_results") or []
                combined_rows = result.get("raw_results") or []

                tab_ctx, tab_models = st.tabs(
                    ["KG / Embedding Context", "Model Answers"]
                )

                with tab_ctx:
                    st.markdown("#### 📂 Combined Context (after de-duplication)")
                    df_combined = format_results_as_dataframe(combined_rows)
                    if not df_combined.empty:
                        st.dataframe(df_combined, use_container_width=True, hide_index=True)
                    else:
                        st.info("No combined context rows.")

                    # Small summary
                    st.markdown(
                        f"- Baseline rows: **{len(baseline_rows)}**  \n"
                        f"- Embedding rows: **{len(emb_rows)}**  \n"
                        f"- Combined rows (after de-dup): **{len(combined_rows)}**"
                    )

                    # Optional detailed views
                    if baseline_rows:
                        with st.expander("Baseline-only rows"):
                            df_b = format_results_as_dataframe(baseline_rows)
                            st.dataframe(df_b, use_container_width=True, hide_index=True)

                    if emb_rows:
                        with st.expander("Embeddings-only rows"):
                            # include score if present
                            df_e = format_results_as_dataframe(emb_rows)
                            st.dataframe(df_e, use_container_width=True, hide_index=True)

                with tab_models:
                    st.markdown("#### 🧠 LLM Answers (using the combined context)")
                    models = result.get("models", {})
                    if not models:
                        st.info("No LLM outputs available.")
                    else:
                        cols = st.columns(len(models))
                        for col, (model_name, info) in zip(cols, models.items()):
                            with col:
                                st.markdown(f"**{model_name}**")
                                if "error" in info:
                                    st.error(info["error"])
                                    continue
                                st.markdown("**Answer:**")
                                st.write(info["answer"])
                                st.markdown("**Quantitative metrics:**")
                                rt = info.get("response_time")
                                if rt is not None:
                                    st.write(f"- Response time: {rt:.3f} s")
                                else:
                                    st.write("- Response time: N/A")
                                st.write(
                                    f"- Prompt tokens: {info.get('prompt_tokens', 'N/A')}"
                                )
                                st.write(
                                    f"- Completion tokens: {info.get('completion_tokens', 'N/A')}"
                                )
                                st.write(
                                    f"- Total tokens: {info.get('total_tokens', 'N/A')}"
                                )
                                st.write(
                                    f"- Estimated cost (USD): {info.get('cost_usd', 'N/A')}"
                                )

    elif submit_button and not query:
        st.warning("⚠️ Please enter a query first!")

    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 1rem;'>
            <p>FPL Knowledge Graph Query Interface | Built with Streamlit</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
