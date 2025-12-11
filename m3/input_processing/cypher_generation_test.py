# test_query_templates.py

import pytest
from cypher_generation import test_cypher_generation3


# ==============================
# Expected template per query
# ==============================

EXPECTED_TEMPLATE_BY_QUERY = {
    # ---- core_tests: PLAYER_PERFORMANCE ----
    "How many points did Mohamed Salah score in 2022-23":
        "player_performance_stats",
    "Show me Erling Haaland's stats for gameweek 10 in 2022-23":
        "player_performance_by_gw",
    "What was Bukayo Saka's total points this season":
        "player_performance_stats",

    # ---- core_tests: PLAYER_RECOMMENDATION ----
    "Who are the best defenders to pick in GW5":
        "top_players_by_position",
    "Recommend top 5 midfielders for gameweek 12":
        "top_players_by_position",
    "Which forwards should I choose for this season":
        "top_players_by_position",
    "Top forwards":
        "top_players_by_position",

    # ---- core_tests: PLAYER_SEARCH / STATS ----
    "Show me all players":
        "all_players_list",
    "List all the players":
        "all_players_list",
    "List all the players in Arsenal in gameweek 5 in 2021-22":
        "players_in_team",
    "What is the average points per game for all players":
        "avg_points_per_game_all_players",

    # ---- core_tests: TEAM_QUERY ----
    "Which teams played in the 2022-23 season":
        "teams_in_season",
    "List all teams":
        "all_teams_list",
    "List all teams in 2022-23":
        "teams_in_season",
    "Find team Arsenal":
        "team_performance",

    # ---- core_tests: FIXTURE_QUERY ----
    "Show me all fixtures for Arsenal":
        "team_fixtures",
    "List all fixtures":
        "all_fixtures_list",
    "Show me Arsenal fixtures for gameweek 10 in 2022-23":
        "team_fixtures",

    # ---- core_tests: GAMEWEEK_QUERY ----
    "What games are in gameweek 5":
        "gameweek_fixtures",
    "What games are in gameweek 5 of 2022-23":
        "gameweek_fixtures",
    "How many total gameweeks are there":
        "total_gameweeks",
    "List all gameweeks":
        "all_gameweeks_list",

    # ---- core_tests: SEASON_QUERY ----
    "List all seasons":
        "all_seasons_list",
    "How many seasons are in the database":
        "all_seasons_list",
    "Show me overall stats for the 2022-23 season":
        "season_overall_stats",

    # ---- core_tests: COMPARISON_QUERY ----
    "Compare Mohamed Salah vs Erling Haaland this season":
        "player_comparison",
    "Compare Mohamed Salah vs Erling Haaland in 2022-23 in gameweek 10":
        "player_comparison",
    "Who scored more points, Harry Kane or Heung-min Son, in 2021-22":
        "player_comparison",

    # ---- core_tests: POSITION_QUERY ----
    "Find players who play as defender":
        "players_by_position",
    "List all positions":
        "all_positions_list",
    "Show me players playing as midfielder":
        "players_by_position",

    # ---- core_tests: ENTITY_SEARCH ----
    "Search for Mohamed Salah":
        "search_players",

    # ---- test_queries_1 ----
    "Who are the top 10 defenders in the 2022-23 season":
        "top_players_by_position",
    "Show me arsenal fixtures for gameweek 10 in 2022-23":
        "team_fixtures",

    # ---- test_queries_2 ----
    "What is the highest points scored by a player":
        "top_players_overall",

    # ---- test_queries_3 ----
    "List all fixtures in gameweek 5 in 2022-23":
        "gameweek_fixtures",

    # ---- test_queries_4 ----
    "List all the player in arsenal in gameweek 5 in 2021-22":
        "players_in_team",

    # ---- edge_tests ----
    "Show me all fixtures where Mohamed Salah played in 2022-23":
        "player_performance_stats",
    "Show me Arsenal's gameweek 10 fixture in 2022-23":
        "team_fixtures",
    "List fixtures for gameweek 3 of 2021-22":
        "gameweek_fixtures",
    "Show me all players who played for Arsenal in 2022-23":
        "players_in_team",
    "Show me all home fixtures for Liverpool in 2022-23":
        "team_fixtures",
    "Show me away fixtures for Chelsea in gameweek 1 of 2022-23":
        "team_fixtures",
    "List all defenders in Manchester City":
        "players_by_position",
    "Which season had the most total goals":
        "season_with_most_total_goals",
    "Show me stats for Arsenal":
        "team_performance",
}

UNKNOWN_QUERIES = [
    "Hello",
    "I love football",
]


# ==============================
# Tests
# ==============================

@pytest.mark.parametrize(
    "query,expected_template",
    list(EXPECTED_TEMPLATE_BY_QUERY.items()),
)
def test_query_template_selection(query: str, expected_template: str) -> None:
    """
    For each known FPL query, the router should return
    the expected Cypher query template key.
    """
    template = test_cypher_generation3(query)
    assert template == expected_template, (
        f"For query: {query!r} expected template "
        f"{expected_template!r} but got {template!r}"
    )


@pytest.mark.parametrize("query", UNKNOWN_QUERIES)
def test_unknown_queries_have_no_template(query: str) -> None:
    """
    Non-FPL / casual queries should not map to any template.
    Adjust this if you represent 'no template' differently.
    """
    template = test_cypher_generation3(query)
    assert template is None or template == "", (
        f"Expected no template for unknown query {query!r}, got {template!r}"
    )
