from neo4j import GraphDatabase
import pandas as pd

CSV_PATH = "fpl_two_seasons.csv"  # <- update this to your CSV file path

def load_config(path="config.txt"):
    config = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if "=" in line:
                key, value = line.split("=", 1)
                config[key] = value.strip().replace('"', '')
    return config


cfg = load_config()

URI = cfg["URI"]
USER = cfg["USERNAME"]
PASSWORD = cfg["PASSWORD"]
DATABASE = "neo4j"


def test_connection():
    driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

    # open a session and run a simple query
    with driver.session(database=DATABASE) as session:
        result = session.run("RETURN 'Connected to Neo4j!' AS msg")
        record = result.single()
        print(record["msg"])

    driver.close()


def create_season_nodes():
    # 1) Read the CSV
    df = pd.read_csv(CSV_PATH)

    # 2) Get unique season values
    unique_seasons = df["season"].unique()
    print("Seasons found in CSV:", unique_seasons)

    # 3) Connect to Neo4j and create Season nodes
    with GraphDatabase.driver(URI, auth=(USER, PASSWORD)) as driver:
        with driver.session(database=DATABASE) as session:
            for season_name in unique_seasons:
                session.run(
                    """
                    MERGE (s:Season {season_name: $season_name})
                    """,
                    season_name=season_name,
                )
    print("Season nodes created (or matched) successfully.")


def create_gameweek_nodes_and_Linking_GW_To_Season_Batch():
    df = pd.read_csv(CSV_PATH)

    # 1. PREPARE DATA
    # We only need the unique pairs of Season + Gameweek
    unique_pairs = df[["season", "GW"]].drop_duplicates()
    
    # Convert to list of dictionaries
    all_records = unique_pairs.to_dict('records')
    
    # 2. DEFINE BATCH SIZE
    # Even though Gameweeks are few (380 per decade), we keep the pattern 
    BATCH_SIZE = 1000 
    total_records = len(all_records)
    
    print(f"Processing {total_records} Gameweeks in batches of {BATCH_SIZE}...")

    query = """
    UNWIND $batch AS row
    
    // 1. Find the Season
    MATCH (s:Season {season_name: row.season})
    
    // 2. Create the Gameweek
    // Note: We use row.season again here to ensure the Gameweek has the season property
    MERGE (gw:Gameweek {season: row.season, GW_number: toInteger(row.GW)})
    
    // 3. Connect them
    MERGE (s)-[:HAS_GW]->(gw)
    """

    with GraphDatabase.driver(URI, auth=(USER, PASSWORD)) as driver:
        with driver.session(database=DATABASE) as session:
            
            # 3. BATCH LOOP
            for i in range(0, total_records, BATCH_SIZE):
                batch_chunk = all_records[i : i + BATCH_SIZE]
                
                print(f"Sending batch {i} to {i + len(batch_chunk)}...")
                session.run(query, batch=batch_chunk)

    print("Gameweek nodes created and linked to Seasons successfully.")


def create_fixture_nodes_and_Linking_GW_To_Fixture_Batch():
    df = pd.read_csv(CSV_PATH)

    # 1. PANDAS: PREPARE DATA
    # Drop duplicates based on the identity keys (season + fixture)
    unique_fixtures = df[["season", "fixture", "GW", "kickoff_time"]].drop_duplicates(subset=["season", "fixture"])
    
    # CONVERSION: Turn the DataFrame into a list of dictionaries (Maps)
    # Example format: [{'season': '2023-24', 'fixture': 1, 'GW': 1, 'kickoff_time': '...'}, {...}]
    fixture_batch = unique_fixtures.to_dict('records')
    
    print(f"Batch processing {len(fixture_batch)} unique fixtures...")

    query = """
    // 2. UNWIND THE BATCH
    // This turns the list of maps back into individual rows inside the database engine
    UNWIND $batch AS row

    // 3. FIND THE GAMEWEEK
    MATCH (gw:Gameweek {season: row.season, GW_number: toInteger(row.GW)})

    // 4. CREATE/MERGE THE FIXTURE
    // Identity: Season + Fixture Number
    MERGE (f:Fixture {season: row.season, fixture_number: toInteger(row.fixture)})
    
    // 5. SET ATTRIBUTE
    // We update the kickoff_time regardless of whether the node was just created or matched
    SET f.kickoff_time = row.kickoff_time

    // 6. CREATE RELATIONSHIP
    MERGE (gw)-[:HAS_FIXTURE]->(f)
    """

    with GraphDatabase.driver(URI, auth=(USER, PASSWORD)) as driver:
        with driver.session(database=DATABASE) as session:
            # We pass the whole list as a single parameter called 'batch'
            session.run(query, batch=fixture_batch)

    print("Fixture nodes created and connected to Gameweeks successfully.")


def create_team_nodes():
    df = pd.read_csv(CSV_PATH)

    # Collect unique team names from both home_team and away_team
    home_teams = df["home_team"].unique()
    away_teams = df["away_team"].unique()

    # Combine and get unique names
    all_teams = set(list(home_teams) + list(away_teams))

    print("Teams found:", all_teams)

    with GraphDatabase.driver(URI, auth=(USER, PASSWORD)) as driver:
        with driver.session(database=DATABASE) as session:
            for team_name in all_teams:
                session.run(
                    """
                    MERGE (t:Team {name: $team_name})
                    """,
                    team_name=team_name
                )

    print("Team nodes created successfully.")


def create_player_nodes_batch():
    df = pd.read_csv(CSV_PATH)

    # 1. PREPARE DATA
    # We drop duplicates based on name AND element to ensure unique combos
    unique_players = df[["element", "name"]].drop_duplicates(subset=["element", "name"])
    
    player_batch = unique_players.to_dict('records')
    
    print(f"Creating {len(player_batch)} unique Player nodes...")

    query = """
    UNWIND $batch AS row
    
    // We MERGE on BOTH element and name to avoid 'ID Recycling' collisions
    MERGE (p:Player {player_element: toInteger(row.element), player_name: row.name})
    """

    with GraphDatabase.driver(URI, auth=(USER, PASSWORD)) as driver:
        with driver.session(database=DATABASE) as session:
            session.run(query, batch=player_batch)

    print("Player nodes created successfully.")


def create_team_nodes():
    df = pd.read_csv(CSV_PATH)

    # Collect unique team names from both home_team and away_team
    home_teams = df["home_team"].unique()
    away_teams = df["away_team"].unique()

    # Combine and get unique names
    all_teams = set(list(home_teams) + list(away_teams))

    print("Teams found:", all_teams)

    with GraphDatabase.driver(URI, auth=(USER, PASSWORD)) as driver:
        with driver.session(database=DATABASE) as session:
            for team_name in all_teams:
                session.run(
                    """
                    MERGE (t:Team {name: $team_name})
                    """,
                    team_name=team_name
                )

    print("Team nodes created successfully.")

def create_position_nodes():
    df = pd.read_csv(CSV_PATH)

    # Get unique positions
    unique_positions = df["position"].dropna().unique()
    print("Positions found:", unique_positions)

    with GraphDatabase.driver(URI, auth=(USER, PASSWORD)) as driver:
        with driver.session(database=DATABASE) as session:
            for pos_name in unique_positions:
                session.run(
                    """
                    MERGE (pos:Position {name: $pos_name})
                    """,
                    pos_name=pos_name
                )

    print("Position nodes created successfully.")



def link_fixtures_to_teams_batch():
    df = pd.read_csv(CSV_PATH)

    # 1. PANDAS: PREPARE DATA
    # We select the necessary columns.
    # We drop duplicates based on the FIXTURE IDENTITY (season + fixture).
    # This ensures we only process each match once.
    unique_matches = df[["season", "fixture", "home_team", "away_team"]].drop_duplicates(subset=["season", "fixture"])
    
    # CONVERSION: Turn DataFrame into a list of dictionaries
    match_batch = unique_matches.to_dict('records')
    
    print(f"Batch linking teams for {len(match_batch)} matches...")

    query = """
    UNWIND $batch AS row

    // 1. FIND THE FIXTURE
    // Critical: We match using 'fixture_number' as defined in your schema.
    MATCH (f:Fixture {season: row.season, fixture_number: toInteger(row.fixture)})

    // 2. FIND/CREATE TEAMS
    // We use MERGE for teams to ensure they are created if they don't exist yet.
    MERGE (home:Team {name: row.home_team})
    MERGE (away:Team {name: row.away_team})

    // 3. CREATE RELATIONSHIPS
    MERGE (f)-[:HAS_HOME_TEAM]->(home)
    MERGE (f)-[:HAS_AWAY_TEAM]->(away)
    """

    with GraphDatabase.driver(URI, auth=(USER, PASSWORD)) as driver:
        with driver.session(database=DATABASE) as session:
            session.run(query, batch=match_batch)

    print("Fixtures successfully linked to Home and Away Teams.")


def link_players_to_positions_batch():
    df = pd.read_csv(CSV_PATH)

    # 1. PANDAS: PREPARE DATA
    # We need the Player Identity (element + name) and their Position.
    # We drop duplicates based on the Player Identity to ensure we process each player only once.
    # We also drop rows where 'position' might be missing (NaN).
    unique_rows = df[["element", "name", "position"]].drop_duplicates()
    
    # CONVERSION: Turn DataFrame into a list of dictionaries
    data_batch = unique_rows.to_dict('records')
    
    print(f"Batch linking {len(data_batch)} players to positions...")

    query = """
    UNWIND $batch AS row
    
    // 1. FIND THE PLAYER
    // We use the exact schema you defined: [player_name, player_element]
    // using MATCH implies the player nodes must already exist.
    MATCH (p:Player {player_element: toInteger(row.element), player_name: row.name})
    
    // 2. FIND/CREATE THE POSITION
    // We use MERGE. If "Midfielder" exists, we match it. If not, we create it.
    MERGE (pos:Position {name: row.position})
    
    // 3. CREATE RELATIONSHIP
    MERGE (p)-[:PLAYS_AS]->(pos)
    """

    with GraphDatabase.driver(URI, auth=(USER, PASSWORD)) as driver:
        with driver.session(database=DATABASE) as session:
            session.run(query, batch=data_batch)

    print("PLAYS_AS relationships created successfully.")



def link_players_to_fixtures_with_stats_batch():
    df = pd.read_csv(CSV_PATH)
    
    # 1. DEFINE STATS COLUMNS
    # We separate them by type so we can cast them correctly in Cypher
    int_stats = [
        'minutes', 'goals_scored', 'assists', 'total_points', 'bonus',
        'clean_sheets', 'goals_conceded', 'own_goals', 'penalties_saved',
        'penalties_missed', 'yellow_cards', 'red_cards', 'saves','bps'
    ]
    
    float_stats = [
        'influence', 'creativity', 'threat', 'ict_index', 'form'
    ]

    all_stat_cols = int_stats + float_stats

    # 2. PANDAS: PREPARE DATA
    # We need the Keys (element, name, season, fixture) + Stats
    cols_to_keep = ["element", "name", "season", "fixture","position"] + all_stat_cols
    
    # Fill NaN values with 0 (essential for math operations later)
    df[all_stat_cols] = df[all_stat_cols].fillna(0)
    
    # Convert to list of dictionaries
    all_records = df[cols_to_keep].to_dict('records')
    
    # 3. DEFINE BATCH SIZE
    BATCH_SIZE = 2000 
    total_records = len(all_records)
    
    print(f"Linking players to fixtures. Processing {total_records} records in batches...")

    # 4. CYPHER QUERY
    # We map the stats explicitly to ensure 'minutes' is an Integer and 'form' is a Float
    query = """
    UNWIND $batch AS row
    
    // 1. MATCH PLAYER
    // Using your specific schema: [player_name, player_element]
    MATCH (p:Player {player_name: row.name, player_element: toInteger(row.element)})
    
    // 2. MATCH FIXTURE
    // Using your specific schema: [season, fixture_number]
    MATCH (f:Fixture {season: row.season, fixture_number: toInteger(row.fixture)})

    // 3. CREATE RELATIONSHIP
    MERGE (p)-[r:PLAYED_IN]->(f)
    
    // 4. SET PROPERTIES
    // We use SET r += {...} to update many properties efficiently
    SET r += {
        minutes: toInteger(row.minutes),
        goals_scored: toInteger(row.goals_scored),
        assists: toInteger(row.assists),
        total_points: toInteger(row.total_points),
        bonus: toInteger(row.bonus),
        clean_sheets: toInteger(row.clean_sheets),
        goals_conceded: toInteger(row.goals_conceded),
        own_goals: toInteger(row.own_goals),
        penalties_saved: toInteger(row.penalties_saved),
        penalties_missed: toInteger(row.penalties_missed),
        yellow_cards: toInteger(row.yellow_cards),
        red_cards: toInteger(row.red_cards),
        saves: toInteger(row.saves),
        bps: toInteger(row.bps),
        
        // Float properties
        influence: toFloat(row.influence),
        creativity: toFloat(row.creativity),
        threat: toFloat(row.threat),
        ict_index: toFloat(row.ict_index),
        form: toFloat(row.form),
        position: row.position
    }
    """

    with GraphDatabase.driver(URI, auth=(USER, PASSWORD)) as driver:
        with driver.session(database=DATABASE) as session:
            
            # 5. EXECUTE IN BATCHES
            for i in range(0, total_records, BATCH_SIZE):
                batch_chunk = all_records[i : i + BATCH_SIZE]
                print(f"Processing batch {i} to {i + len(batch_chunk)}...")
                session.run(query, batch=batch_chunk)

    print("All PLAYED_IN relationships with full stats created successfully.")




if __name__ == "__main__":
    test_connection()
    create_season_nodes()
    create_gameweek_nodes_and_Linking_GW_To_Season_Batch()
    create_fixture_nodes_and_Linking_GW_To_Fixture_Batch()
    create_team_nodes()
    create_player_nodes_batch()
    create_position_nodes() 
    link_fixtures_to_teams_batch()
    link_players_to_positions_batch()
    link_players_to_fixtures_with_stats_batch()






