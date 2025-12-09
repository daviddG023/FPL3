# FPL Query UI - Streamlit Interface

This is the Streamlit UI implementation for Part 4 of Milestone 3.

## Features

- 🎨 **FPL-themed interface** with green color scheme
- 💬 **Natural language query input**
- 📊 **Intent classification display**
- 🔍 **Entity extraction visualization**
- 📋 **Results displayed as interactive tables**
- 📥 **CSV download functionality**
- 📝 **Query history**
- 💡 **Quick example queries**

## Installation

Make sure you have the required packages installed:

```bash
pip install streamlit pandas neo4j
```

## Running the Application

From the project root directory, run:

```bash
streamlit run m3/fpl_query_ui.py
```

Or if you're in the m3 directory:

```bash
streamlit run fpl_query_ui.py
```

The application will open in your default web browser at `http://localhost:8501`

## Usage

1. Enter your query in natural language (e.g., "How many points did Mohamed Salah score in 2022-23?")
2. Click the "Query" button or press Enter
3. View the detected intent and extracted entities
4. See the results displayed in an interactive table
5. Download results as CSV if needed

## Example Queries

- "How many points did Mohamed Salah score in 2022-23?"
- "Who are the top 10 defenders in 2022-23?"
- "Show me Arsenal fixtures for gameweek 10"
- "Compare Mohamed Salah vs Erling Haaland"
- "What games are in gameweek 5?"
- "Find players who play as defender"
- "Who scored the most goals in 2022-23?"

## Configuration

The application uses the configuration file at `m2/config.txt` for Neo4j database connection. Make sure this file exists and contains valid credentials.

## Notes

- The UI automatically connects to the Neo4j database on startup
- Query history is maintained during the session
- Results are limited to 20 rows in the display (full results can be downloaded)

