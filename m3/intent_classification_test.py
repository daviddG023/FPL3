from intent_classification import Intent, IntentClassifier

intent_classifier = IntentClassifier()

core_tests = [
    # PLAYER_PERFORMANCE
    ("How many points did Mohamed Salah score in 2022-23?", Intent.PLAYER_PERFORMANCE),
    ("Show me Erling Haaland's stats for gameweek 10 in 2022-23", Intent.PLAYER_PERFORMANCE),
    ("What was Bukayo Saka's total points this season?", Intent.PLAYER_PERFORMANCE),

    # PLAYER_RECOMMENDATION
    ("Who are the best defenders to pick in GW5?", Intent.PLAYER_RECOMMENDATION),
    ("Recommend top 5 midfielders for gameweek 12", Intent.PLAYER_RECOMMENDATION),
    ("Which forwards should I choose for this season?", Intent.PLAYER_RECOMMENDATION),
    ("Top forwards", Intent.PLAYER_RECOMMENDATION),

    # PLAYER_SEARCH
    ("Show me all players", Intent.PLAYER_SEARCH),
    ("List all the players", Intent.PLAYER_SEARCH),
    ("List all the players in Arsenal in gameweek 5 in 2021-22", Intent.PLAYER_SEARCH),
    ("What is the average points per game for all players?", Intent.PLAYER_SEARCH), 

    # TEAM_QUERY
    ("Which teams played in the 2022-23 season?", Intent.TEAM_QUERY),
    ("List all teams", Intent.TEAM_QUERY),
    ("List all teams in 2022-23", Intent.TEAM_QUERY),
    ("Find team Arsenal", Intent.TEAM_QUERY), 

    # FIXTURE_QUERY
    ("Show me all fixtures for Arsenal", Intent.FIXTURE_QUERY),
    ("List all fixtures", Intent.FIXTURE_QUERY),
    ("Show me Arsenal fixtures for gameweek 10 in 2022-23", Intent.FIXTURE_QUERY),

    # GAMEWEEK_QUERY
    ("What games are in gameweek 5?", Intent.GAMEWEEK_QUERY),
    ("What games are in gameweek 5 of 2022-23?", Intent.GAMEWEEK_QUERY),
    ("How many total gameweeks are there?", Intent.GAMEWEEK_QUERY),
    ("List all gameweeks", Intent.GAMEWEEK_QUERY),

    # SEASON_QUERY
    ("List all seasons", Intent.SEASON_QUERY),
    ("How many seasons are in the database?", Intent.SEASON_QUERY),
    ("Show me overall stats for the 2022-23 season", Intent.SEASON_QUERY),

    # COMPARISON_QUERY
    ("Compare Mohamed Salah vs Erling Haaland this season", Intent.COMPARISON_QUERY),
    ("Compare Mohamed Salah vs Erling Haaland in 2022-23 in gameweek 10", Intent.COMPARISON_QUERY),
    ("Who scored more points, Harry Kane or Heung-min Son, in 2021-22?", Intent.COMPARISON_QUERY),

    # POSITION_QUERY
    ("Find players who play as defender", Intent.POSITION_QUERY),
    ("List all positions", Intent.POSITION_QUERY),
    ("Show me players playing as midfielder", Intent.POSITION_QUERY),

    # ENTITY_SEARCH
    ("Search for Mohamed Salah", Intent.ENTITY_SEARCH),
    

    # UNKNOWN
    ("Hello", Intent.UNKNOWN),
    ("I love football", Intent.UNKNOWN),
]

test_queries_1 = [
    ("How many points did Mohamed Salah score in 2022-23?", Intent.PLAYER_PERFORMANCE),
    ("Who are the top 10 defenders in the 2022-23 season?", Intent.PLAYER_RECOMMENDATION),
    ("Show me arsenal fixtures for gameweek 10 in 2022-23", Intent.FIXTURE_QUERY),
    ("What games are in gameweek 5 of 2022-23?", Intent.GAMEWEEK_QUERY),
    ("Compare Mohamed Salah vs Erling Haaland in 2022-23 in gameweek 10", Intent.COMPARISON_QUERY),
    ("Show me Erling Haaland's stats for gameweek 10 in 2022-23", Intent.PLAYER_PERFORMANCE),
]

test_queries_2 = [
    ("How many points did Mohamed Salah score in 2022-23?", Intent.PLAYER_PERFORMANCE),
    ("Who are the best defenders to pick in GW5?", Intent.PLAYER_RECOMMENDATION),
    ("Show me all fixtures for Arsenal", Intent.FIXTURE_QUERY),
    ("What games are in gameweek 5?", Intent.GAMEWEEK_QUERY),
    ("Which teams played in the 2022-23 season?", Intent.TEAM_QUERY),
    ("Compare Mohamed Salah vs Erling Haaland this season", Intent.COMPARISON_QUERY),
    ("Find players who play as defender", Intent.POSITION_QUERY),
    ("How many total gameweeks are there?", Intent.GAMEWEEK_QUERY),
    ("What is the highest points scored by a player?", Intent.PLAYER_RECOMMENDATION), 
    ("Show me all players", Intent.PLAYER_SEARCH),
]

test_queries_3 = [
    ("List all teams", Intent.TEAM_QUERY),
    ("List all positions", Intent.POSITION_QUERY),
    ("List all seasons", Intent.SEASON_QUERY),
    ("List all gameweeks", Intent.GAMEWEEK_QUERY),
    ("List all fixtures", Intent.FIXTURE_QUERY),
    ("List all teams in 2022-23", Intent.TEAM_QUERY),
    ("List all fixtures in gameweek 5 in 2022-23", Intent.GAMEWEEK_QUERY),
]

test_queries_4 = [
    ("List all the player in arsenal in gameweek 5 in 2021-22", Intent.PLAYER_SEARCH),
    ("Top forwards", Intent.PLAYER_RECOMMENDATION), 

]

edge_tests = [
    # Player + fixture-ish words but no explicit GW
    ("Show me all fixtures where Mohamed Salah played in 2022-23", Intent.PLAYER_PERFORMANCE),
    # (or PLAYER_SEARCH depending on how you treat "played in" + fixtures)

    # Team + GW + Season, no player
    ("Show me Arsenal's gameweek 10 fixture in 2022-23", Intent.FIXTURE_QUERY),

    # Pure fixture with season and gw
    ("List fixtures for gameweek 3 of 2021-22", Intent.GAMEWEEK_QUERY),

    # Player + team (multi-entity, no stats words)
    ("Show me all players who played for Arsenal in 2022-23", Intent.PLAYER_SEARCH),

    # Home vs away team (still fixture intent)
    ("Show me all home fixtures for Liverpool in 2022-23", Intent.FIXTURE_QUERY),
    ("Show me away fixtures for Chelsea in gameweek 1 of 2022-23", Intent.FIXTURE_QUERY),

    # Position + team filter
    ("List all defenders in Manchester City", Intent.POSITION_QUERY),  # or PLAYER_SEARCH + filter

    # Season-level stats vs intent
    ("Which season had the most total goals?", Intent.SEASON_QUERY), 

    # Vague but domain-y
    ("Show me stats for Arsenal", Intent.TEAM_QUERY),  # you could later introduce TEAM_STATS if you want
]

def run_tests(classifier: IntentClassifier, tests):
    failed_tests = []

    for q, expected in tests:
        intent, entities = classifier.classify(q)  # or however your API looks
        if intent == expected :
            status = "OK"
        else:
            status = f"FAIL (got {intent})" 
            failed_tests.append(q)
            print(f"[{status}] {q}\n  expected: {expected}, got: {intent}\n")
    print(f"Failed tests: {failed_tests}")
all_tests = core_tests + test_queries_1 + test_queries_2 + test_queries_3 + test_queries_4 + edge_tests
run_tests(intent_classifier, all_tests)
