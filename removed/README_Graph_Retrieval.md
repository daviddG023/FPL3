# Graph Retrieval Layer - Baseline Implementation

This document describes the Graph Retrieval Layer implementation for the FPL (Fantasy Premier League) knowledge graph system.

## Overview

The Graph Retrieval Layer provides a baseline approach to retrieve information from the Neo4j knowledge graph using structured Cypher queries. It integrates with the Intent Classification system to automatically select and parameterize queries based on user intents and extracted entities.

## Components

### 1. GraphRetrieval Class (`Graph_Retrieval.py`)

The main class that handles:

- Neo4j database connection
- Query template library (14+ templates)
- Query selection based on intent
- Entity-based parameterization
- Query execution and result formatting

### 2. Query Templates

The system includes **14 query templates** covering different intents:

#### Player Performance (2 queries)

- `player_performance_stats`: Get performance statistics for a specific player
- `player_performance_by_gw`: Get player performance for a specific gameweek

#### Player Recommendations (2 queries)

- `top_players_by_position`: Get top players by position in a season
- `top_players_overall`: Get top players overall by total points

#### Fixture Queries (1 query)

- `team_fixtures`: Get all fixtures for a specific team

#### Gameweek Queries (1 query)

- `gameweek_fixtures`: Get all fixtures in a specific gameweek

#### Season Queries (1 query)

- `teams_in_season`: Get all teams that played in a season

#### Comparison Queries (1 query)

- `player_comparison`: Compare statistics between multiple players

#### Position Queries (1 query)

- `players_by_position`: Get all players who play in a specific position

#### Statistics Queries (3 queries)

- `total_gameweeks`: Get total number of gameweeks
- `top_scorers`: Get top goal scorers
- `top_assists`: Get players with most assists

#### Entity Search (1 query)

- `search_players`: Search for players by name (partial match)

#### Team Queries (1 query)

- `team_performance`: Get team performance statistics

## Usage

### Basic Usage

```python
from intent_classification_chat import IntentClassifier
from Graph_Retrieval import GraphRetrieval

# Initialize components
classifier = IntentClassifier()

# Load Neo4j config
config = load_config("m2/config.txt")
retrieval = GraphRetrieval(
    uri=config["URI"],
    username=config["USERNAME"],
    password=config["PASSWORD"],
    database="neo4j"
)

# Process a query
query = "How many points did Mohamed Salah score in 2022-23?"

# Step 1: Classify intent and extract entities
intent, metadata = classifier.classify(query)
entities = metadata["entities"]

# Step 2: Retrieve from graph
results = retrieval.retrieve(intent.value.upper(), entities)

# Step 3: Process results
for record in results:
    print(record)
```

### Complete Example

See `retrieval_example.py` for a complete working example that demonstrates:

- Intent classification
- Query selection
- Parameterization
- Execution
- Result formatting

## Query Selection Logic

The system uses a scoring mechanism to select the most appropriate query:

1. **Filter by Intent**: Only queries matching the classified intent are considered
2. **Check Required Entities**: Queries without required entities are skipped
3. **Score by Entity Match**:
   - Required entities: +10 points each
   - Optional entities: +5 points each
4. **Select Best Match**: Query with highest score is selected

## Entity Parameterization

The system automatically extracts and maps entities to query parameters:

- **Players**: `player_name` (single) or `player_names` (list for comparisons)
- **Teams**: `team_name`
- **Seasons**: `season` and `season_name` (both formats supported)
- **Gameweeks**: `gw_number` (converted to integer)
- **Positions**: `position` (normalized: GK, DEF, MID, FWD)
- **Stats**: Used for filtering (if applicable)

### Position Mapping

Common position names are automatically normalized:

- `gk`, `goalkeeper` → `GK`
- `def`, `defender` → `DEF`
- `mid`, `midfielder` → `MID`
- `fwd`, `forward`, `attacker` → `FWD`

## Query Templates Structure

Each query template includes:

```python
{
    "intent": "PLAYER_PERFORMANCE",  # Intent category
    "description": "Get performance statistics...",  # Human-readable description
    "template": "MATCH ... RETURN ...",  # Cypher query template
    "required_entities": ["players", "seasons"],  # Required entities
    "optional_entities": [],  # Optional entities
    "default_params": {"limit": 10}  # Default parameters (optional)
}
```

## Example Queries and Their Mappings

| User Query                                            | Intent                | Selected Query             | Entities Used                                               |
| ----------------------------------------------------- | --------------------- | -------------------------- | ----------------------------------------------------------- |
| "How many points did Mohamed Salah score in 2022-23?" | PLAYER_PERFORMANCE    | `player_performance_stats` | players: ["Mohamed Salah"], seasons: ["2022-23"]            |
| "Top 10 defenders in 2022-23"                         | PLAYER_RECOMMENDATION | `top_players_by_position`  | positions: ["DEF"], seasons: ["2022-23"]                    |
| "Arsenal fixtures in gameweek 10"                     | FIXTURE_QUERY         | `team_fixtures`            | teams: ["Arsenal"], seasons: ["2022-23"], gameweeks: ["10"] |
| "Compare Salah vs Haaland"                            | COMPARISON_QUERY      | `player_comparison`        | players: ["Mohamed Salah", "Erling Haaland"]                |
| "Who scored most goals in 2022-23?"                   | STATISTICS_QUERY      | `top_scorers`              | seasons: ["2022-23"]                                        |

## Integration with Intent Classifier

The Graph Retrieval Layer is designed to work seamlessly with the Intent Classifier:

1. **Intent Classification**: `IntentClassifier.classify()` returns intent and entities
2. **Query Selection**: `GraphRetrieval.select_query()` finds matching template
3. **Parameterization**: `GraphRetrieval.parameterize_query()` fills in parameters
4. **Execution**: `GraphRetrieval.execute_query()` runs the Cypher query

## Error Handling

The system handles various error cases:

- **No matching query**: Returns empty results if no suitable template found
- **Missing entities**: Skips queries that require entities not present
- **Database errors**: Catches and reports Neo4j connection/query errors
- **Invalid parameters**: Validates parameters before execution

## Configuration

The system requires Neo4j connection details in `m2/config.txt`:

```
URI=bolt://localhost:7687
USERNAME=neo4j
PASSWORD=your_password
DATABASE=neo4j
```

## Future Enhancements

The baseline implementation can be extended with:

1. **Embedding-based Retrieval**: Add semantic similarity search (Section 2.b)
2. **Query Optimization**: Cache frequently used queries
3. **Result Ranking**: Score and rank results by relevance
4. **Multi-query Fusion**: Combine results from multiple queries
5. **Natural Language Generation**: Convert results to natural language responses

## Files

- `Graph_Retrieval.py`: Main retrieval class with query templates
- `intent_classification_chat.py`: Intent classifier with entity extraction
- `retrieval_example.py`: Complete usage example
- `README_Graph_Retrieval.md`: This documentation

## Testing

Run the example script to test the system:

```bash
python "m3/Input Preprocessing/retrieval_example.py"
```

Or test individual components:

```bash
# Test query templates
python "m3/Input Preprocessing/Graph_Retrieval.py"

# Test intent classification
python "m3/Input Preprocessing/intent_classification_chat.py"
```
