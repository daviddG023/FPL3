"""
Graph Retrieval Layer for FPL Knowledge Graph

This module implements Cypher query templates and retrieval logic to fetch
relevant information from the Neo4j knowledge graph based on user intents and
extracted entities.

Features:
- Baseline retrieval using structured Cypher queries
- At least 10 query templates covering different intents
- Entity-based parameterization
- Query execution and result formatting
"""

from neo4j import GraphDatabase
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
import json


class GraphRetrieval:
    """
    Graph Retrieval Layer for executing Cypher queries against the FPL KG.
    
    This class provides:
    1. Connection to Neo4j database
    2. Query template library (10+ templates)
    3. Query selection based on intent
    4. Entity-based parameterization
    5. Query execution and result formatting
    """
    
    def __init__(self, uri: str, username: str, password: str, database: str = "neo4j"):
        """
        Initialize the Graph Retrieval layer.
        
        Args:
            uri: Neo4j database URI
            username: Neo4j username
            password: Neo4j password
            database: Database name (default: "neo4j")
        """
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.database = database
        self.query_templates = self._initialize_query_templates()
    
    def close(self):
        """Close the Neo4j driver connection"""
        if self.driver:
            self.driver.close()
    
    def _initialize_query_templates(self) -> Dict[str, Dict[str, Any]]:
        """
        Initialize the library of Cypher query templates.
        
        Returns:
            Dictionary mapping query names to their templates and metadata
        """
        return {
            # Query 1: Player Performance - Get specific player stats
            "player_performance_stats": {
                "intent": "PLAYER_PERFORMANCE",
                "description": "Get performance statistics for a specific player",
                "template": """
                    MATCH (p:Player {player_name: $player_name})-[r:PLAYED_IN]->(f:Fixture)
                    WHERE coalesce($season, f.season) = f.season
                    OPTIONAL MATCH (f)-[:HAS_HOME_TEAM|HAS_AWAY_TEAM]->(t:Team)
                    WITH p, r, f, collect(DISTINCT t.name) as teams
                    RETURN p.player_name as player_name,
                           p.player_element as player_element,
                           sum(r.minutes) as total_minutes,
                           sum(r.goals_scored) as total_goals,
                           sum(r.assists) as total_assists,
                           sum(r.total_points) as total_points,
                           sum(r.clean_sheets) as total_clean_sheets,
                           avg(r.form) as avg_form,
                           avg(r.ict_index) as avg_ict_index,
                           count(r) as appearances
                    ORDER BY total_points DESC
                """,
                "required_entities": ["players"],
                "optional_entities": []
            },
            

            
            # Query 2: Player Performance by Gameweek
            "player_performance_by_gw": {
                "intent": "PLAYER_PERFORMANCE",
                "description": "Get player performance for a specific gameweek",
                "template": """
                    MATCH (p:Player {player_name: $player_name})-[r:PLAYED_IN]->(f:Fixture)
                    MATCH (gw:Gameweek)-[:HAS_FIXTURE]->(f)
                    WHERE ($season IS NULL OR gw.season = $season)
                    AND ($gw_number IS NULL OR gw.GW_number = $gw_number)
                    OPTIONAL MATCH (f)-[:HAS_HOME_TEAM]->(home:Team)
                    OPTIONAL MATCH (f)-[:HAS_AWAY_TEAM]->(away:Team)
                    RETURN p.player_name as player_name,
                           gw.GW_number as gameweek,
                           f.fixture_number as fixture,
                           r.minutes as minutes,
                           r.goals_scored as goals,
                           r.assists as assists,
                           r.total_points as points,
                           r.bonus as bonus,
                           r.clean_sheets as clean_sheets,
                           home.name as home_team,
                           away.name as away_team
                    ORDER BY f.fixture_number
                """,
                "required_entities": ["players", "seasons", "gameweeks"],
                "optional_entities": []
            },
            
            # Query 3: Top Players by Position
            "top_players_by_position": {
                "intent": "PLAYER_RECOMMENDATION",
                "description": "Get top players by position",
                "template": """
                    MATCH (p:Player)-[:PLAYS_AS]->(pos:Position {name: $position})
                    MATCH (p)-[r:PLAYED_IN]->(f:Fixture)
                    WHERE coalesce($season, f.season) = f.season
                    MATCH (gw:Gameweek)-[:HAS_FIXTURE]->(f)
                    WHERE ($season IS NULL OR gw.season = $season)
                    AND ($gw_number IS NULL OR gw.GW_number = $gw_number)
                    WITH p, pos, sum(r.total_points) as total_points,
                         sum(r.goals_scored) as total_goals,
                         sum(r.assists) as total_assists,
                         sum(r.minutes) as total_minutes,
                         count(r) as appearances,
                         avg(r.form) as avg_form,
                         avg(r.ict_index) as avg_ict_index,
                         gw.GW_number as gameweek
                    WHERE appearances > 0
                    RETURN p.player_name as player_name,
                           pos.name as position,
                           total_points,
                           total_goals,
                           total_assists,
                           total_minutes,
                           appearances,
                           round(avg_form, 2) as avg_form,
                           round(avg_ict_index, 2) as avg_ict_index,
                           gameweek
                    ORDER BY total_points DESC
                    LIMIT $limit
                """,
                "required_entities": ["positions"],
                "optional_entities": ["seasons", "gameweeks"],
                "default_params": {"limit": 10}
            },
            
            # Query 4: Top Players Overall (All Positions)
            "top_players_overall": {
                "intent": "PLAYER_RECOMMENDATION",
                "description": "Get top players overall by total points",
                "template": """
                    MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)
                    WHERE coalesce($season, f.season) = f.season
                    MATCH (p)-[:PLAYS_AS]->(pos:Position)
                    WITH p, collect(DISTINCT pos.name) as positions,
                         sum(r.total_points) as total_points,
                         sum(r.goals_scored) as total_goals,
                         sum(r.assists) as total_assists,
                         count(r) as appearances
                    WHERE appearances > 0
                    RETURN p.player_name as player_name,
                           positions,
                           total_points,
                           total_goals,
                           total_assists,
                           appearances
                    ORDER BY total_points DESC
                    LIMIT $limit
                """,
                "required_entities": [],
                "optional_entities": [],
                "default_params": {"limit": 10}
            },
            
            # Query 5: Team Fixtures
            "team_fixtures": {
                "intent": "FIXTURE_QUERY",
                "description": "Get all fixtures for a specific team",
                "template": """
                    MATCH (f:Fixture)-[:HAS_HOME_TEAM|HAS_AWAY_TEAM]->(t:Team {name: $team_name})
                    WHERE coalesce($season, f.season) = f.season
                    OPTIONAL MATCH (f)-[:HAS_HOME_TEAM]->(home:Team)
                    OPTIONAL MATCH (f)-[:HAS_AWAY_TEAM]->(away:Team)
                    MATCH (gw:Gameweek)-[:HAS_FIXTURE]->(f)
                    WHERE ($gw_number IS NULL OR gw.GW_number = $gw_number)
                    AND ($season IS NULL OR gw.season = $season OR f.season = $season)

                    RETURN f.fixture_number as fixture_number,
                           f.kickoff_time as kickoff_time,
                           home.name as home_team,
                           away.name as away_team,
                           gw.GW_number as gameweek
                    ORDER BY f.fixture_number
                """,
                "required_entities": ["teams"],
                "optional_entities": ["seasons", "gameweeks"]
            },
            
            # Query 5b: Gameweek Fixtures (when gameweek is specified but no team)
            "gameweek_fixtures_simple": {
                "intent": "FIXTURE_QUERY",
                "description": "Get all fixtures in a gameweek",
                "template": """
                    MATCH (gw:Gameweek)-[:HAS_FIXTURE]->(f:Fixture)
                    WHERE ($gw_number IS NULL OR gw.GW_number = $gw_number)
                    AND ($season IS NULL OR gw.season = $season OR f.season = $season)
                    MATCH (gw)-[:HAS_FIXTURE]->(f:Fixture)
                    OPTIONAL MATCH (f)-[:HAS_HOME_TEAM]->(home:Team)
                    OPTIONAL MATCH (f)-[:HAS_AWAY_TEAM]->(away:Team)
                    RETURN gw.GW_number as gameweek,
                           f.fixture_number as fixture_number,
                           f.kickoff_time as kickoff_time,
                           home.name as home_team,
                           away.name as away_team
                    ORDER BY f.kickoff_time
                """,
                "required_entities": ["gameweeks"],
                "optional_entities": ["seasons"]
            },
            
            # Query 6: Gameweek Fixtures
            "gameweek_fixtures": {
                "intent": "GAMEWEEK_QUERY",
                "description": "Get all fixtures in a specific gameweek",
                "template": """
                    MATCH (gw:Gameweek)-[:HAS_FIXTURE]->(f:Fixture)
                    WHERE ($season IS NULL OR gw.season = $season OR f.season = $season)
                    AND ($gw_number IS NULL OR gw.GW_number = $gw_number)

                    MATCH (gw)-[:HAS_FIXTURE]->(f:Fixture)
                    OPTIONAL MATCH (f)-[:HAS_HOME_TEAM]->(home:Team)
                    OPTIONAL MATCH (f)-[:HAS_AWAY_TEAM]->(away:Team)
                    RETURN gw.GW_number as gameweek,
                           f.fixture_number as fixture_number,
                           f.kickoff_time as kickoff_time,
                           home.name as home_team,
                           away.name as away_team
                    ORDER BY f.kickoff_time
                """,
                "required_entities": ["gameweeks"],
                "optional_entities": ["seasons"]
            },
            
            # Query 7: Teams in Season
            "teams_in_season": {
                "intent": "SEASON_QUERY",
                "description": "Get all teams that played in a season",
                "template": """
                    MATCH (gw:Gameweek)-[:HAS_FIXTURE]->(f:Fixture)
                    WHERE ($season IS NULL OR gw.season = $season OR f.season = $season)
                    AND ($gw_number IS NULL OR gw.GW_number = $gw_number)
                    MATCH (f)-[:HAS_HOME_TEAM|HAS_AWAY_TEAM]->(t:Team)
                    RETURN DISTINCT t.name as team_name
                    ORDER BY team_name
                """,
                "required_entities": [],
                "optional_entities": ["seasons"]
            },
            
            # Query 7b: Total Gameweeks
            "total_gameweeks": {
                "intent": "STATISTICS_QUERY",
                "description": "Get total number of gameweeks",
                "template": """
                    MATCH (gw:Gameweek)-[:HAS_FIXTURE]->(f:Fixture)
                    WHERE ($season IS NULL OR gw.season = $season OR f.season = $season)
                    AND ($gw_number IS NULL OR gw.GW_number = $gw_number)
                    RETURN count(DISTINCT gw) as total_gameweeks
                """,
                "required_entities": [],
                "optional_entities": ["seasons"]
            },
            
            # Query 7c: Highest Points by Player
            "highest_points_player": {
                "intent": "PLAYER_RECOMMENDATION",
                "description": "Get player with highest points",
                "template": """
                    MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)
                    WHERE coalesce($season, f.season) = f.season
                    WITH p, max(r.total_points) as max_points
                    MATCH (p)-[r2:PLAYED_IN]->(f2:Fixture)
                    WHERE coalesce($season, f2.season) = f2.season AND r2.total_points = max_points
                    RETURN p.player_name as player_name,
                           r2.total_points as points,
                           f2.fixture_number as fixture,
                           f2.season as season
                    ORDER BY max_points DESC
                    LIMIT 1
                """,
                "required_entities": [],
                "optional_entities": []
            },
            
            # Query 8: Player Comparison
            "player_comparison": {
                "intent": "COMPARISON_QUERY",
                "description": "Compare statistics between multiple players",
                "template": """
                    MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)
                    WHERE p.player_name IN $player_names AND coalesce($season, f.season) = f.season
                    MATCH (gw:Gameweek)-[:HAS_FIXTURE]->(f)
                    WHERE ($season IS NULL OR gw.season = $season)
                    AND ($gw_number IS NULL OR gw.GW_number = $gw_number)
                    WITH p, 
                         sum(r.total_points) as total_points,
                         sum(r.goals_scored) as total_goals,
                         sum(r.assists) as total_assists,
                         sum(r.minutes) as total_minutes,
                         sum(r.clean_sheets) as total_clean_sheets,
                         count(r) as appearances,
                         avg(r.form) as avg_form,
                         avg(r.ict_index) as avg_ict_index,
                         max(r.total_points) as best_game_points
                    RETURN p.player_name as player_name,
                           total_points,
                           total_goals,
                           total_assists,
                           total_minutes,
                           total_clean_sheets,
                           appearances,
                           round(avg_form, 2) as avg_form,
                           round(avg_ict_index, 2) as avg_ict_index,
                           best_game_points
                    ORDER BY total_points DESC
                """,
                "required_entities": ["players"],
                "optional_entities": ["seasons", "gameweeks"]
            },

            # Query 9: Players by Position
            "players_by_position": {
                "intent": "POSITION_QUERY",
                "description": "Get all players who play in a specific position",
                "template": """
                    MATCH (p:Player)-[:PLAYS_AS]->(pos:Position {name: $position})
                    OPTIONAL MATCH (p)-[r:PLAYED_IN]->(f:Fixture)
                    WHERE coalesce($season, f.season) = f.season
                    WITH p, pos, 
                         sum(r.total_points) as total_points,
                         count(r) as appearances
                    RETURN p.player_name as player_name,
                           p.player_element as player_element,
                           pos.name as position,
                           total_points,
                           appearances
                    ORDER BY total_points DESC
                """,
                "required_entities": ["positions"],
                "optional_entities": ["seasons", "gameweeks"]
            },
            
            
            # Query 11: Statistics - Top Scorers
            "top_scorers": {
                "intent": "STATISTICS_QUERY",
                "description": "Get top goal scorers",
                "template": """
                    MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)
                    WHERE coalesce($season, f.season) = f.season
                    WITH p, sum(r.goals_scored) as total_goals,
                         sum(r.total_points) as total_points,
                         count(r) as appearances
                    WHERE total_goals > 0
                    RETURN p.player_name as player_name,
                           total_goals,
                           total_points,
                           appearances
                    ORDER BY total_goals DESC, total_points DESC
                    LIMIT $limit
                """,
                "required_entities": [],
                "optional_entities": ["seasons"],
                "default_params": {"limit": 10}
            },
            
            # Query 12: Statistics - Most Assists
            "top_assists": {
                "intent": "STATISTICS_QUERY",
                "description": "Get players with most assists",
                "template": """
                    MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)
                    WHERE coalesce($season, f.season) = f.season
                    WITH p, sum(r.assists) as total_assists,
                         sum(r.total_points) as total_points,
                         count(r) as appearances
                    WHERE total_assists > 0
                    RETURN p.player_name as player_name,
                           total_assists,
                           total_points,
                           appearances
                    ORDER BY total_assists DESC, total_points DESC
                    LIMIT $limit
                """,
                "required_entities": [],
                "optional_entities": ["seasons"],
                "default_params": {"limit": 10}
            },
            
            # Query 13: Entity Search - Search Players by Name
            "search_players": {
                "intent": "ENTITY_SEARCH",
                "description": "Search for players by name (partial match)",
                "template": f"""
                    MATCH (p:Player{{player_name: $player_name}})
                    RETURN p.player_name as player_name,
                           p.player_element as player_element
                    LIMIT $limit
                """,
                "required_entities": ["players"],
                "optional_entities": [],
                "default_params": {"limit": 20}
            },
            
            # Query 14: Team Analysis - Team Performance Stats
            "team_performance": {
                "intent": "TEAM_QUERY",
                "description": "Get team performance statistics",
                "template": """
                    MATCH (f:Fixture)-[:HAS_HOME_TEAM|HAS_AWAY_TEAM]->(t:Team {name: $team_name})
                    WHERE coalesce($season, f.season) = f.season
                    MATCH (gw:Gameweek)-[:HAS_FIXTURE]->(f)
                    WHERE ($season IS NULL OR gw.season = $season)
                    AND ($gw_number IS NULL OR gw.GW_number = $gw_number)
                    OPTIONAL MATCH (f)-[:HAS_HOME_TEAM]->(home:Team)
                    OPTIONAL MATCH (f)-[:HAS_AWAY_TEAM]->(away:Team)
                    OPTIONAL MATCH (p:Player)-[r:PLAYED_IN]->(f)
                    WHERE (home.name = $team_name OR away.name = $team_name)
                    WITH t, f, home, away,
                         sum(CASE WHEN home.name = $team_name THEN 1 ELSE 0 END) as home_matches,
                         sum(CASE WHEN away.name = $team_name THEN 1 ELSE 0 END) as away_matches,
                         sum(r.total_points) as team_total_points,
                         sum(r.goals_scored) as team_goals_scored
                    RETURN t.name as team_name,
                           count(DISTINCT f) as total_fixtures,
                           home_matches,
                           away_matches,
                           team_total_points,
                           team_goals_scored
                """,
                "required_entities": ["teams"],
                "optional_entities": ["seasons", "gameweeks"]
            },
            # Query 15 List all players
            "all_players_list": {
                "intent": "PLAYER_SEARCH",
                "description": "List all players",
                "template": """
                    MATCH (p:Player)
                    RETURN p.player_name AS player_name,
                           p.player_element AS player_element
                    ORDER BY player_name
                    LIMIT $limit
                """,
                "required_entities": [],
                "optional_entities": [],
                "default_params": {"limit": 200}
            },
            # Query 16 List all teams
            "all_teams_list": {
                "intent": "TEAM_QUERY",
                "description": "List all teams",
                "template": """
                    MATCH (t:Team)
                    RETURN t.name AS team_name
                """,
                "required_entities": [],
                "optional_entities": [],
            },
            # Query 17 List all positions
            "all_positions_list": {
                "intent": "POSITION_QUERY",
                "description": "List all positions",
                "template": """
                    MATCH (p:Position)
                    RETURN p.name AS position
                """,
                "required_entities": [],
                "optional_entities": [],
            },
            # Query 18 List all seasons
            "all_seasons_list": {
                "intent": "SEASON_QUERY",
                "description": "List all seasons",
                "template": """
                    MATCH (s:Season)
                    RETURN s.season_name AS season_name
                """,
                "required_entities": [],
                "optional_entities": [],
            },
            # Query 19 List all gameweeks
            "all_gameweeks_list": {
                "intent": "GAMEWEEK_QUERY",
                "description": "List all gameweeks",
                "template": """
                    MATCH (gw:Gameweek)
                    RETURN gw.GW_number AS gameweek
                """,
                "required_entities": [],
                "optional_entities": [],
            },
            # Query 20 List all fixtures
            "all_fixtures_list": {
                "intent": "FIXTURE_QUERY",
                "description": "List all fixtures",
                "template": """
                    MATCH (f:Fixture)
                    RETURN f.fixture_number AS fixture_number
                """,
                "required_entities": [],
                "optional_entities": [],
            },
            # Query 21 List all teams in a season
            "teams_in_season": {
                "intent": "TEAM_QUERY",
                "description": "List all teams in a season",
                "template": """
                    MATCH (t:Team)-[r:PLAYED_IN]->(f:Fixture)
                    WHERE r.season = $season
                """,
                "required_entities": ["seasons"],
                "optional_entities": [],
            },
            # Query 22 List all fixtures in a gameweek
            "fixtures_in_gameweek": {
                "intent": "GAMEWEEK_QUERY",
                "description": "List all fixtures in a gameweek",
                "template": """
                    MATCH (f:Fixture)-[r:PLAYED_IN]->(gw:Gameweek)
                    WHERE r.gameweek = $gameweek
                """,
                "required_entities": ["gameweeks"],
                "optional_entities": [],
            },
            # Query 23 List all players in a team
            "players_in_team": {
                "intent": "PLAYER_SEARCH",
                "description": "List all players in a team",
                "template": """

                    MATCH (p:Player)-[:PLAYED_IN]->(f:Fixture)
                    MATCH (f)-[:HAS_HOME_TEAM|HAS_AWAY_TEAM]->(T:Team {name: $team_name})
                    WITH DISTINCT p,T
                    MATCH (gw:Gameweek)-[:HAS_FIXTURE]->(f)
                    WHERE ($season IS NULL OR gw.season = $season)
                    AND ($gw_number IS NULL OR gw.GW_number = $gw_number)
                    // Step 2: Exclude players who EVER appeared in a non-team_name team
                    AND NOT EXISTS {
                        MATCH (p)-[:PLAYED_IN]->(f2:Fixture)
                        MATCH (f2)-[:HAS_HOME_TEAM]->(home:Team)
                        MATCH (f2)-[:HAS_AWAY_TEAM]->(away:Team)
                        WHERE home.name <> $team_name AND away.name <> $team_name
                    }

                    RETURN DISTINCT p.player_name AS player_name, T.name AS team_name, 
                    gw.GW_number AS gameweek,gw.season AS season
                    ORDER BY player_name;
                """,
                "required_entities": ["teams"],
                "optional_entities": ["seasons", "gameweeks"],
            },
            # Query 24 Average points per game for all players
            "avg_points_per_game_all_players": {
                "intent": "PLAYER_SEARCH",
                "description": "Compute the average points per game across all players in the database",
                "template": """
                    MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)
                    WITH p,
                        sum(r.points) AS totalPoints,
                        count(f)      AS gamesPlayed
                    WHERE gamesPlayed > 0
                    RETURN avg(1.0 * totalPoints / gamesPlayed) AS avg_points_per_game;
                """,
                # as you said: both must be present
                "required_entities": ["stat_ops", "stats"],
                "optional_entities": []
            },

            #Query 25 Show overall stats for the 2022-23 season
            "season_overall_stats": {
                "intent": "SEASON_QUERY",
                "description": "Show overall aggregated stats (points, goals, assists) for a given season",
                "template": """
                    MATCH (s:Season {season_name: $season})
                        -[:HAS_GW]->(gw:Gameweek)
                        -[:HAS_FIXTURE]->(f:Fixture)
                    MATCH (p:Player)-[r:PLAYED_IN]->(f)
                    RETURN 
                        s.season_name AS season,
                        sum(r.points)  AS total_points,
                        sum(r.goals)   AS total_goals,
                        sum(r.assists) AS total_assists;
                """,
                # needs a season, and we know it's an aggregate because stat_ops is present
                "required_entities": ["seasons", "stat_ops"],
                "optional_entities": ["stats"]
            },

            # Query 26 Which season had the most total goals?
            "season_with_most_total_goals": {
                "intent": "SEASON_QUERY",
                "description": "Find the season with the highest total number of goals scored",
                "template": """
                    MATCH (s:Season)
                        -[:HAS_GW]->(gw:Gameweek)
                        -[:HAS_FIXTURE]->(f:Fixture)
                    MATCH (p:Player)-[r:PLAYED_IN]->(f)
                    WITH s, sum(r.goals) AS total_goals
                    RETURN 
                        s.season_name AS season,
                        total_goals
                    ORDER BY total_goals DESC
                    LIMIT 1;
                """,
                # no specific season, but we *need* stat_ops + stats for this
                "required_entities": ["stat_ops", "stats"],
                "optional_entities": []
            },
            # Query 27 How many total gameweeks are there?
            "total_gameweeks": {
                "intent": "GAMEWEEK_QUERY",
                "description": "Get total number of gameweeks",
                "template": """
                    MATCH (gw:Gameweek)-[:HAS_FIXTURE]->(f:Fixture)
                    RETURN count(DISTINCT gw) as total_gameweeks
                """,
                "required_entities": ["stat_ops"],
                "optional_entities": []
            },



        }
    
    def select_query(self, intent: str, entities: Dict[str, List[str]]) -> Optional[str]:
        """
        Select the most appropriate query template based on intent and available entities.
        
        Args:
            intent: The classified intent (e.g., "PLAYER_PERFORMANCE" or "player_performance")
            entities: Dictionary of extracted entities
            
        Returns:
            Query template name or None if no suitable query found
        """
        # Normalize intent to match template format (uppercase)
        intent_upper = intent.upper()
        
        # Filter queries by intent
        candidate_queries = {
            name: template 
            for name, template in self.query_templates.items()
            if template["intent"] == intent_upper
        }
        
        if not candidate_queries:
            return None
        
        # Score each candidate query based on available entities
        best_query = None
        best_score = -1
        
        for query_name, template in candidate_queries.items():
            score = 0
            required = template.get("required_entities", [])
            optional = template.get("optional_entities", [])
            
            # Check required entities
            has_all_required = True
            for req_entity in required:
                if req_entity in entities and len(entities[req_entity]) > 0:
                    score += 10  # High weight for required entities
                else:
                    has_all_required = False
                    break
            
            if not has_all_required:
                continue  # Skip queries without required entities
            
            # Bonus for optional entities
            for opt_entity in optional:
                if opt_entity in entities and len(entities[opt_entity]) > 0:
                    score += 5
            
            if score > best_score:
                best_score = score
                best_query = query_name
        
        return best_query
    

    def parameterize_query(self, query_name: str, entities: Dict[str, List[str]], 
                          additional_params: Optional[Dict] = None) -> Tuple[str, Dict]:
        """
        Parameterize a query template with extracted entities.
        Uses coalesce pattern to handle missing seasons/gameweeks.
        
        Args:
            query_name: Name of the query template
            entities: Extracted entities from the query
            additional_params: Additional parameters to include
            
        Returns:
            Tuple of (parameterized_query_string, parameters_dict)
        """
        if query_name not in self.query_templates:
            raise ValueError(f"Query template '{query_name}' not found")
        
        template_info = self.query_templates[query_name]
        query_template = template_info["template"]
        default_params = template_info.get("default_params", {})
        
        
        # Build parameters dictionary
        params = {}
        params.update(default_params)
        
        if additional_params:
            params.update(additional_params)
        
        # Extract entity values
        has_season = "seasons" in entities and len(entities["seasons"]) > 0
        has_gameweek = "gameweeks" in entities and len(entities["gameweeks"]) > 0
        
        if "players" in entities and len(entities["players"]) > 0:
            # Clean player names (remove possessive 's)
            cleaned_players = [p.replace("'s", "").strip() for p in entities["players"]]
            params["player_name"] = cleaned_players[0]  # Use first player
            params["player_names"] = cleaned_players  # For comparison queries
        
        if "teams" in entities and len(entities["teams"]) > 0:
            params["team_name"] = entities["teams"][0]  # Use first team
        
        if has_season:
            params["season"] = entities["seasons"][0]  # Use first season
            params["season_name"] = entities["seasons"][0]  # Some queries use season_name
        else:
            # Set to None so coalesce will match all
            params["season"] = None
            params["season_name"] = None
        
        if "gameweeks" in entities and len(entities["gameweeks"]) > 0:
            params["gw_number"] = int(entities["gameweeks"][0])  # Use first gameweek
        else:
            params["gw_number"] = None
        
        if "positions" in entities and len(entities["positions"]) > 0:
            # Map common position names to standard format
            position_map = {
                "gk": "GK", "goalkeeper": "GK",
                "def": "DEF", "defender": "DEF",
                "mid": "MID", "midfielder": "MID",
                "fwd": "FWD", "forward": "FWD", "attacker": "FWD"
            }
            pos = entities["positions"][0].lower()
            params["position"] = position_map.get(pos, pos.upper())
        
        if "stats" in entities and len(entities["stats"]) > 0:
            # Stats can be used for filtering, but most queries don't need this
            pass
        
        return query_template, params
    


    def generate_cypher_query(self, intent: str, entities: Dict[str, List[str]], 
                              additional_params: Optional[Dict] = None) -> Tuple[Optional[str], str, Dict]:
        """
        Generate the actual Cypher query string with entities filled in.
        
        Args:
            intent: Classified intent (will be normalized to uppercase)
            entities: Extracted entities
            additional_params: Additional parameters (e.g., limit, offset)
            
        Returns:
            Tuple of (query_name, cypher_query_string, parameters_dict)
            Returns (None, "", {}) if no suitable query found
        """
        # Normalize intent to uppercase for matching
        intent_upper = intent.upper()
        
        # Select appropriate query
        query_name = self.select_query(intent_upper, entities)
        
        if not query_name:
            return None, "", {}
        
        # Parameterize query
        query_template, params = self.parameterize_query(query_name, entities, additional_params)
        
        # Replace parameters in query template to show actual values
        # Note: This is for display purposes - actual execution uses parameterized queries
        cypher_query = query_template
        
        # Process lists first to avoid partial replacements
        list_params = {k: v for k, v in params.items() if isinstance(v, list)}
        for param_name, param_value in list_params.items():
            # Format list as Cypher list (for IN clauses)
            list_str = "[" + ", ".join(f'"{v}"' if isinstance(v, str) else str(v) for v in param_value) + "]"
            # Replace IN clause pattern first
            cypher_query = cypher_query.replace(f"IN ${param_name}", f"IN {list_str}")
            # Then replace standalone parameter
            cypher_query = cypher_query.replace(f"${param_name}", list_str)
        
        # Process other parameters
        for param_name, param_value in params.items():
            if param_name in list_params:
                continue  # Already processed
            if isinstance(param_value, str):
                cypher_query = cypher_query.replace(f"${param_name}", f'"{param_value}"')
            else:
                cypher_query = cypher_query.replace(f"${param_name}", str(param_value))
        cypher_query = cypher_query.replace("None", "null")

        return query_name, cypher_query, params
    


    def get_query_info(self, query_name: str) -> Dict[str, Any]:
        """
        Get information about a query template.
        
        Args:
            query_name: Name of the query template
            
        Returns:
            Dictionary with query information
        """
        if query_name not in self.query_templates:
            return {}
        
        template = self.query_templates[query_name].copy()
        # Remove the actual template string for cleaner output
        template.pop("template", None)
        return template


    def execute_query(self, query: str, parameters: Dict) -> List[Dict[str, Any]]:
        """
        Execute a Cypher query against the Neo4j database.
        
        Args:
            query: Cypher query string
            parameters: Query parameters dictionary
            
        Returns:
            List of result records as dictionaries
        """
        results = []
        
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, parameters)
                
                for record in result:
                    # Convert Neo4j record to dictionary
                    record_dict = {}
                    for key in record.keys():
                        value = record[key]
                        # Handle Neo4j-specific types
                        if hasattr(value, '__iter__') and not isinstance(value, (str, bytes)):
                            try:
                                value = list(value)
                            except:
                                pass
                        record_dict[key] = value
                    results.append(record_dict)
        
        except Exception as e:
            print(f"Error executing query: {e}")
            print(f"Query: {query}")
            print(f"Parameters: {parameters}")
            raise
        
        return results
    