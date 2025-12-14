import re
from typing import Dict, List, Tuple, Optional
from enum import Enum


class Intent(Enum):
    PLAYER_PERFORMANCE = "player_performance"
    PLAYER_RECOMMENDATION = "player_recommendation"
    PLAYER_SEARCH = "player_search"
    TEAM_QUERY = "team_query"
    FIXTURE_QUERY = "fixture_query"
    GAMEWEEK_QUERY = "gameweek_query"
    SEASON_QUERY = "season_query"
    STATISTICS_QUERY = "statistics_query"
    COMPARISON_QUERY = "comparison_query"
    ENTITY_SEARCH = "entity_search"
    POSITION_QUERY = "position_query"
    UNKNOWN = "unknown"


class IntentClassifier:
    """
    Simpler, priority-based intent classifier for FPL queries,
    aligned with your KG:

    Nodes:
      - Season(season_name)
      - Gameweek(season_name, gw_number)
      - Fixture(season_name, fixture_number, kickoff_time)
      - Team(name)
      - Player(player_name, player_element)
      - Position(name)

    Relationships:
      (Season)-[:HAS_GW]->(Gameweek)
      (Gameweek)-[:HAS_FIXTURE]->(Fixture)
      (Fixture)-[:HAS_HOME_TEAM]->(Team)
      (Fixture)-[:HAS_AWAY_TEAM]->(Team)
      (Player)-[:PLAYS_AS]->(Position)
      (Player)-[:PLAYED_IN]->(Fixture {stats...})
    """

    def __init__(self):
        # --- keyword groups ---

        # these are mostly for detecting entity type
        self.player_words = {
            'forward', 'midfielder',
            'defender', 'goalkeeper', 'gk', 'def', 'mid', 'fwd'
        }

        self.team_words = {
            'arsenal', 'chelsea', 'liverpool',
            'manchester', 'city', 'united', 'tottenham', 'spurs', 'leicester',
            'brighton', 'crystal palace', 'everton', 'fulham', 'leeds', 'newcastle',
            'southampton', 'west ham', 'wolves', 'aston villa', 'brentford',
            'burnley', 'norwich', 'watford', 'sheffield', 'bournemouth'
        }#'team', 'teams', 'club', 'clubs',

        self.fixture_words = {
            'fixture', 'fixtures', 'match', 'matches', 'game ', 'games ',
            'kickoff', 'kickoff time', 'against'
        }

        self.gameweek_words = {
            'gameweek', 'game week', 'gw', 'week', 'round','gameweeks'
        }

        self.season_words = {
            # "season", "seasons",
            "2021-22", "2022-23", "2023-24", "2024-25"
        }

        self.position_words = {
            'gk', 'goalkeeper', 'def', 'defender',
            'mid', 'midfielder', 'fwd', 'forward', 'attacker'
        }#'position', 'positions',

        self.stat_words = {
            "points", "point", "goals", "goal", "assists", "assist",
            "clean sheet", "clean sheets", "yellow card", "yellow cards",
            "red card", "red cards", "saves", "bonus", "bps",
            "influence", "creativity", "threat", "ict", "ict index",
            "minutes", "mins", "appearances", "form"
        }

        # --- action / intent indicators ---
        self.recommend_words = {
            "top","best", "recommend", "recommendation", "pick", "picks", "should i", "who should", "who should i", "which player", "suggest", "advice", "who to"
        }

        self.compare_words = {
            "compare", "comparison", "vs", "versus", "better", "worse", "than",'difference',
            'between',  'versus', "more", "less"
        }

        self.aggregate_words = {
            "how many", "how much", "total", "count", "sum",
            "average", "avg", "mean", "most", "least", "highest", "lowest",
            "total", "count", "sum","overall","average"
        }
        self.stat_ops_map = {
            "average": "avg", "avg": "avg", "mean": "avg",
            "total": "sum", "sum": "sum", "overall": "sum",
            "how many": "count", "count": "count", "number of": "count",
            "highest": "max", "most": "max", "maximum": "max", "top": "max",
            "lowest": "min", "least": "min", "minimum": "min",
        }

        self.search_words = {
            "find", "search", "show", "list", "get", "display"
        }


    def classify(self, query: str) -> Tuple[Intent, Dict[str, any]]: 
        """
        Main entry point.

        Returns:
            intent: Intent enum
            metadata: dict with entities, flags, etc.
        """
        original_query = query.strip()
        q = original_query.lower()

        entities = self._extract_entities(original_query)
        flags = self._extract_flags(q)

        intent = self._decide_intent(q, entities, flags)

        metadata = {
            "entities": entities,
            "flags": flags
        }

        return intent, metadata

    # ------------------------------------------------------------------
    # Step 1: basic feature extraction
    # ------------------------------------------------------------------

    def _extract_flags(self, q: str) -> Dict[str, bool]:
        """Binary features: what kind of words appear in the query?"""
        def contains_any(words: set) -> bool:
            return any(w in q for w in words)

        flags = {
            "has_player_word": contains_any(self.player_words),
            "has_team_word": contains_any(self.team_words),
            "has_fixture_word": contains_any(self.fixture_words),
            "has_gameweek_word": contains_any(self.gameweek_words),
            "has_season_word": contains_any(self.season_words),
            "has_position_word": contains_any(self.position_words),
            "has_stat_word": contains_any(self.stat_words),
            "has_recommend_word": contains_any(self.recommend_words),
            "has_compare_word": contains_any(self.compare_words),
            "has_aggregate_word": contains_any(self.aggregate_words),
            "has_search_word": contains_any(self.search_words),
            "has_how_many": "how many" in q,
            "has_gw_number": bool(re.search(r"\b(gw|gameweek|game week)\s*(\d+)\b", q)),
            "has_gameweek_domain": contains_any(self.gameweek_words),
        }

        return flags

    def _extract_entities(self, query: str) -> Dict[str, List[str]]:
        q_lower = query.lower()
        entities = {
            "players": [],
            "teams": [],
            "seasons": [],
            "gameweeks": [],
            "fixtures": [],
            "positions": [],
            "stats": [],
            "stat_ops": []
        }

        # ---------- PLAYERS ----------
        # Detect sequences of 2+ capitalised words, but do NOT start
        # a name with common command words like "Compare", "Show", etc.
        stop_starters = {
            "compare", "show", "find", "list", "get",
            "which", "what", "who", "when", "where", "how"
        }

        tokens = query.split()
        players: List[str] = []
        i = 0

        while i < len(tokens):
            raw = tokens[i].strip(",.:;!?")
            # Capitalised word? (e.g. "Mohamed")
            if (
                len(raw) > 1
                and raw[0].isupper()
                and raw[1:].islower()
                and raw.lower() not in stop_starters
            ):
                # start a run of capitalised words
                run = [raw]
                j = i + 1
                while j < len(tokens):
                    raw_j = tokens[j].strip(",.:;!?")
                    if (
                        len(raw_j) > 1
                        and raw_j[0].isupper()
                        and raw_j[1:].islower()
                    ):
                        run.append(raw_j)
                        j += 1
                    else:
                        break

                # we only treat 2+ words as a player name: "Mohamed Salah"
                if len(run) >= 2:
                    name = " ".join(run)
                    if name not in players:
                        players.append(name)

                i = j
            else:
                i += 1
        entities["teams"] = [team.title() for team in self.team_words if team.lower() in q_lower]
        entities["players"] = players

        # ---------- SEASONS ----------
        season_pattern = r"\b(202[1-9]-\d{2})\b"
        entities["seasons"] = re.findall(season_pattern, query)

        # ---------- GAMEWEEKS ----------
        gw_pattern = r"\b(gw|gameweek|gameweeks|game week)\s*(\d+)\b"
        entities["gameweeks"] = [
            m[1] for m in re.findall(gw_pattern, query, re.IGNORECASE)
        ]

        # ---------- FIXTURES ----------
        fixture_pattern = r"\b(fixture|match|game)\s*(?:number|#)?\s*(\d+)\b"
        entities["fixtures"] = [
            m[1] for m in re.findall(fixture_pattern, query, re.IGNORECASE)
        ]

        # ---------- POSITIONS ----------
        q_lower = query.lower()
        entities["positions"] = [
            w for w in self.position_words if w in q_lower
        ]

        # ---------- STATS ----------
        entities["stats"] = [
            w for w in self.stat_words if w in q_lower
        ]
        
        for word, tag in self.stat_ops_map.items():
            if word in q_lower:
                entities["stat_ops"].append(tag)


        entities["stat_ops"] = list(set(entities["stat_ops"]))

        return entities

    # ------------------------------------------------------------------
    # Step 2: rule-based decision
    # ------------------------------------------------------------------
    def _decide_intent(
        self,
        q: str,
        entities: Dict[str, List[str]],
        f: Dict[str, bool],
    ) -> Intent:
        """
        Priority decision system for FPL intent detection.

        Overall priority:
        1. High-value intents (comparison, recommendation)
        2. Player-focused queries (performance, position, player search with filters)
        3. Team + fixtures ⇒ fixtures
        4. Team-focused queries
        5. Gameweek / fixture queries
        6. Season / gameweek generic
        7. General stats
        8. Entity search / fallback
        """
        q_lower = q.lower()

        # Domain flags (semantic domains, not exact keyword matches)
        f["has_player_domain"] = any(
            word in q_lower
            for word in ["player", "players", "scorer", "scorers", "goal scorer", "goal scorers"]
        )
        f["has_team_domain"] = any(word in q_lower for word in ["team", "teams", "clubs", "club"])
        f["has_position_domain"] = any(word in q_lower for word in ["position", "positions"])
        f["has_season_domain"] = any(word in q_lower for word in ["season", "seasons"])

        has_player_name = len(entities["players"]) > 0
        has_team_name = len(entities["teams"]) > 0
        has_season_entity = bool(entities["seasons"])
        has_gw_entity = bool(entities["gameweeks"])

        has_gameweek_context = (
            f["has_gameweek_word"]
            or f["has_gw_number"]
            or f.get("has_gameweek_domain", False)
            or has_gw_entity
        )

        # ======================================================
        # 1. HIGH-PRIORITY INTENTS — always override
        # ======================================================

        # Comparisons ("A vs B", "compare X and Y")
        # if f["has_compare_word"] and (has_player_name or f["has_player_word"] or has_team_name):
        #     return Intent.COMPARISON_QUERY
        # 🔥 STRONG comparison override
        if (
            f["has_compare_word"]
            or (
                has_player_name
                and len(entities["players"]) >= 2
                and any(w in q_lower for w in ["vs", "versus", " or ", " and "])
            )
        ):
            return Intent.COMPARISON_QUERY


        # ======================================================
        # 🔥 1.5 GLOBAL / ANALYTICAL STATISTICS (NEW BLOCK)
        # ======================================================
        # e.g.
        # - "What is the average points per game for all players?"
        # - "What is the highest points scored by a player?"
        # - "Which season had the most total goals?"

        # ======================================================
        # Recommendations ("best", "top", "who should I pick")
        # ======================================================
        # if f["has_recommend_word"] and (has_player_name or f["has_player_word"] or f["has_player_domain"]):
        #     return Intent.PLAYER_RECOMMENDATION
        if (
            f["has_recommend_word"]
            and not f["has_aggregate_word"]
            and not f["has_stat_word"]
        ):
            return Intent.PLAYER_RECOMMENDATION

        # ======================================================
        # 2. PLAYER-FOCUSED INTENTS
        # ======================================================

        # 2a. Player performance
        if has_player_name and (
            f["has_stat_word"]
            or f["has_aggregate_word"]
            or f["has_gameweek_word"]
            or f["has_gw_number"]
            or f["has_season_word"]
            or has_season_entity
        ):
            return Intent.PLAYER_PERFORMANCE
        
        if ((f["has_stat_word"] or f["has_aggregate_word"])
            and not has_player_name
            and not has_team_name
            and not f["has_team_word"]
        ):
            return Intent.STATISTICS_QUERY

        # 2b. Position queries
        if (
            f["has_position_word"]
            or f["has_position_domain"]
            or any(phrase in q_lower for phrase in ["play as", "plays as", "playing as"])
        ):
            return Intent.POSITION_QUERY

        # 2c. Player search WITH filters
        if f["has_player_domain"] and (has_team_name or has_gameweek_context or has_season_entity):
            return Intent.PLAYER_SEARCH

        # 2d. Generic player search
        if f["has_player_domain"]:
            return Intent.PLAYER_SEARCH

        # ======================================================
        # 3. TEAM + FIXTURES ⇒ FIXTURE_QUERY
        # ======================================================
        if (has_team_name or f["has_team_word"] or f["has_team_domain"]) and f["has_fixture_word"]:
            return Intent.FIXTURE_QUERY

        # ======================================================
        # 4. TEAM INTENTS
        # ======================================================
        if has_team_name or f["has_team_word"] or f["has_team_domain"]:
            return Intent.TEAM_QUERY

        # ======================================================
        # 5. GAMEWEEK / FIXTURES (no team/player)
        # ======================================================
        if (
            has_gameweek_context
            and not has_team_name
            and not f["has_team_word"]
            and not has_player_name
            and not f["has_player_word"]
        ):
            return Intent.GAMEWEEK_QUERY

        if f["has_fixture_word"] and not has_gameweek_context:
            return Intent.FIXTURE_QUERY

        # ======================================================
        # 6. SEASON / GAMEWEEK GENERIC
        # ======================================================
        if f["has_season_word"] or has_season_entity or f["has_season_domain"]:
            return Intent.SEASON_QUERY

        if has_gameweek_context or f.get("has_gameweek_domain", False):
            return Intent.GAMEWEEK_QUERY

        # ======================================================
        # 7. GENERAL STATISTICS
        # ======================================================
        if f["has_stat_word"] or f["has_aggregate_word"]:
            return Intent.STATISTICS_QUERY

        # ======================================================
        # 8. ENTITY_SEARCH / FALLBACKS
        # ======================================================
        if f["has_search_word"]:
            return Intent.ENTITY_SEARCH

        if has_player_name or f["has_player_word"]:
            return Intent.ENTITY_SEARCH

        return Intent.UNKNOWN



if __name__ == "__main__":
    classifier = IntentClassifier()

    player_performance_tests = [
    "How many goals did Erling Haaland score?",
    "Mohamed Salah points in 2021-22",
    "Show me assists for Kevin De Bruyne in 2022-23",
    "How many clean sheets did Alisson keep?",
    "What is Haaland's total points this season?",
    "Salah goals in gameweek 10",
    "How many minutes did Bukayo Saka play in 2022-23?",
    ]

    player_recommendation_tests = [
        "Best defenders to pick this season",
        "Top midfielders for GW10",
        "Who should I pick as captain this week?",
        "Which goalkeeper is the best?",
        "Best budget forwards",
        "Top defenders under 5.0",
        "Who should I buy for gameweek 12?",
    ]

    player_search_tests = [
    "Show me all players",
    "List all players in Arsenal",
    "Find players from Liverpool",
    "List all defenders",
    "Show me Arsenal players in 2022-23",
    "Find players in gameweek 5",
]

    comparison_tests = [
        "Compare Salah vs Haaland",
        "Who is better Salah or Haaland?",
        "Saka vs Martinelli",
        "Compare De Bruyne and Bruno Fernandes in 2022-23",
        "Haaland vs Kane points",
    ]

    team_query_tests = [
        "List all teams",
        "Which teams are in the 2022-23 season?",
        "Show me Premier League teams",
        "Teams in 2021-22",
        "Which clubs played in 2022-23?",
    ]

    fixture_query_tests = [
    "Show me Arsenal fixtures",
    "Liverpool fixtures in 2022-23",
    "Fixtures for Manchester City",
    "Show upcoming fixtures for Chelsea",
    "Arsenal vs Tottenham fixture",
]
    
    gameweek_query_tests = [
    "What games are in gameweek 5?",
    "List all fixtures in GW10",
    "How many fixtures are in gameweek 1?",
    "Show me gameweek 20",
    "Which matches are in GW38?",
]
    
    season_query_tests = [
    "List all seasons",
    "Which seasons are available?",
    "What seasons do you have?",
    "Show me all seasons in the database",
]
    
    statistics_query_tests = [
    "What is the average points per player?",
    "Which season had the highest total goals?",
    "How many goals were scored in 2022-23?",
    "What is the total number of players?",
    "How many assists in the 2021-22 season?",
    "What is the highest points scored by a player?",
]
    
    position_query_tests = [
    "List all positions",
    "What positions are there?",
    "Find players who play as midfielder",
    "Who plays as goalkeeper?",
    "Players playing as defender",
]
    
    edge_case_tests = [
    # season + stats
    "Best players in 2022-23",              # recommendation, NOT statistics
    "Top scorers in 2021-22",                # statistics OR recommendation (your design choice)
    
    # gameweek + fixtures
    "Fixtures in gameweek 5",                # gameweek_query
    "Arsenal fixtures in gameweek 5",        # fixture_query
    
    # verbs that look like stats
    "Which teams played last season?",       # team_query
    "Who played in gameweek 1?",              # gameweek_query
    
    # mixed domains
    "Compare Arsenal and Chelsea fixtures",  # comparison_query OR fixture_query (your call)
    
    # short queries
    "GW5",
    "Arsenal fixtures",
    "Best defenders",
]




    test_queries = [
        "How many points did Mohamed Salah score in 2022-23?",
        "Who are the best defenders to pick in GW5?",
        "Show me all fixtures for Arsenal in gameweek 10",
        "What games are in gameweek 5?",
        "Which teams played in the 2022-23 season?",
        "Compare Mohamed Salah vs Erling Haaland this season",
        "Find players who play as defender",
        "How many total gameweeks are there?",
        "What is the highest points scored by a player?",
        "Show me all players",
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
        "List all the players",
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
    ]
    test_queries_5 = [
        'What is the average points per game for all players?',
        'Show me overall stats for the 2022-23 season',
        'Which season had the most total goals?'
    ]
    all_tests = (
    player_performance_tests
    + player_recommendation_tests
    + player_search_tests
    + comparison_tests
    + team_query_tests
    + fixture_query_tests
    + gameweek_query_tests
    + season_query_tests
    + statistics_query_tests
    + position_query_tests
    + edge_case_tests
)

    for q in all_tests:
        intent, meta = classifier.classify(q)
        # hints = classifier.get_cypher_hints(intent)
        print(f"\nQuery: {q}")
        print(f"Intent: {intent.value}")
        print(f"Entities: {meta['entities']}")
        # print(f"Pattern:\n{hints['query_pattern']}")
        print("-" * 60)
