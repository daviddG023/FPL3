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
            'kickoff', 'kickoff time', 'vs', 'versus', 'against'
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
            "minutes", "mins", "played", "appearances", "form"
        }

        # --- action / intent indicators ---
        self.recommend_words = {
            "top","best","highest","most", "recommend", "recommendation", "pick", "picks", "should i", "who should", "who should i", "which player", "suggest", "advice", "who to"
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

        intent = self._decide_intent2(q, entities, flags)

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
        2. Player-focused queries
        3. Team + fixtures ⇒ fixtures
        4. Team-focused queries
        5. Fixture-focused queries
        6. Season / gameweek
        7. General stats
        8. Entity search / fallback
        """
        q_lower = q.lower()
        f["has_player_domain"] = any(word in q_lower for word in ["player", "players",'scorer', 'scorers', 'goal scorer', 'goal scorers'])
        # f["has_gameweek_domain"] = any(word in q_lower for word in ["gameweek", "gw", "game week", "gameweeks"])
        # f["has_fixture_domain"] = any(word in q_lower for word in ["fixture", "fixtures", "match", "matches", "games"])
        f["has_team_domain"]   = any(word in q_lower for word in ["team", "teams", "clubs","club"])
        f["has_position_domain"] = any(word in q_lower for word in ["position", "positions"])
        f["has_season_domain"] = any(word in q_lower for word in ["season", "seasons"])#"season", "seasons",

        has_player_name = len(entities["players"]) > 0
        has_team_name = len(entities["teams"]) > 0
        has_gameweek_context = f["has_gameweek_word"] or f["has_gw_number"] or f["has_gameweek_domain"]
        # ======================================================
        # 1. HIGH-PRIORITY INTENTS — always override
        # ======================================================
        # print(f"f: {f}")
        # Comparisons ("A vs B", "compare X and Y")
        if f["has_compare_word"] and (has_player_name or f["has_player_word"] or has_team_name):
            return Intent.COMPARISON_QUERY

        # Recommendations ("best", "top", "who should I pick")
        if f["has_recommend_word"] and (has_player_name or f["has_player_word"]):
            return Intent.PLAYER_RECOMMENDATION

        # ======================================================
        # 2. PLAYER-FOCUSED (performance / stats / gw / season)
        # ======================================================

        if (has_player_name or f["has_player_word"]) and (
            f["has_stat_word"]
            or f["has_aggregate_word"]
            or f["has_gameweek_word"]
            or f["has_gw_number"]
            or f["has_season_word"]
            or entities["seasons"]
        ):
            return Intent.PLAYER_PERFORMANCE

        # Position queries ("players who play as DEF")
        if f["has_position_word"] or any(w in q for w in ["play as", "plays as", "playing as"])or f["has_position_domain"]:
            return Intent.POSITION_QUERY


        if f["has_player_domain"]:
            return Intent.PLAYER_SEARCH
        # ======================================================
        # 3. TEAM + FIXTURES ⇒ FIXTURE_QUERY
        if (has_team_name or f["has_team_word"]) and (f["has_fixture_word"]):
            return Intent.FIXTURE_QUERY

        # 4. TEAM INTENTS (no explicit fixtures)
        if has_team_name or f["has_team_word"] or f["has_team_domain"]:
            return Intent.TEAM_QUERY

        # 5. PURE GAMEWEEK INTENTS (no specific team / player)
        #    e.g. "What games are in gameweek 5?",
        #         "How many total gameweeks are there?"
        if has_gameweek_context and not has_team_name and not f["has_team_word"] \
        and not has_player_name and not f["has_player_word"]:
            return Intent.GAMEWEEK_QUERY

        # 6. PURE FIXTURE INTENTS (no gameweek context)
        if f["has_fixture_word"] and not has_gameweek_context:
            return Intent.FIXTURE_QUERY

        # 7. SEASON / GAMEWEEK INTENTS (generic)
        if f["has_season_word"] or entities["seasons"] or f["has_season_domain"]:
            return Intent.SEASON_QUERY

        if has_gameweek_context or f["has_gameweek_domain"]:
            return Intent.GAMEWEEK_QUERY

        # 8. GENERAL STATISTICS
        if f["has_stat_word"] or f["has_aggregate_word"]:
            return Intent.STATISTICS_QUERY

        # 9–10. ENTITY_SEARCH / UNKNOWN as you had
        if f["has_search_word"]:
            return Intent.ENTITY_SEARCH

        if has_player_name or f["has_player_word"]:
            return Intent.ENTITY_SEARCH
        return Intent.UNKNOWN      

    def _decide_intent2(
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
        if f["has_compare_word"] and (has_player_name or f["has_player_word"] or has_team_name):
            return Intent.COMPARISON_QUERY

        # Recommendations ("best", "top", "who should I pick")
        if f["has_recommend_word"] and (has_player_name or f["has_player_word"] or f["has_player_domain"]):
            return Intent.PLAYER_RECOMMENDATION
        # is_global_stats = (
        #     (f["has_stat_word"] or f["has_aggregate_word"])  # TRUE
        #     and not has_player_name  # TRUE
        #     and not has_team_name  # TRUE
        #     and not has_gameweek_context  # TRUE
        #     and not has_season_entity  # TRUE
        #     and not f["has_season_word"]  # FALSE - fails here!
        #     and not f.get("has_season_domain", False)  # would also be FALSE
        # )
        # if is_global_stats:
        #     return Intent.STATISTICS_QUERY
        # ======================================================
        # 2. PLAYER-FOCUSED INTENTS
        #    Order:
        #    a) Player performance (needs concrete stat / gw / season)
        #    b) Position queries
        #    c) Player search with filters (team/gw/season)
        #    d) Generic player search
        # ======================================================

        # 2a. Player performance: asks about stats for a specific player over gw/season
        if has_player_name and (
            f["has_stat_word"]
            or f["has_aggregate_word"]
            or f["has_gameweek_word"]
            or f["has_gw_number"]
            or f["has_season_word"]
            or has_season_entity
        ):
            return Intent.PLAYER_PERFORMANCE

        # 2b. Position queries ("players who play as DEF", "list all positions")
        if (
            f["has_position_word"]
            or f["has_position_domain"]
            or any(phrase in q_lower for phrase in ["play as", "plays as", "playing as"])
        ):
            return Intent.POSITION_QUERY

        # 2c. Player search WITH filters (more specific than plain player_search)
        #     e.g. "List all the player in Arsenal",
        #          "List all the player in Arsenal in gameweek 5 in 2021-22"
        if f["has_player_domain"] and (has_team_name or has_gameweek_context or has_season_entity):
            return Intent.PLAYER_SEARCH

        # 2d. Generic player search
        #     e.g. "Show me all players", "List all the players",
        #          "What is the highest points scored by a player?"
        #     (you *want* this to be player_search per your tests)
        if f["has_player_domain"]:
            return Intent.PLAYER_SEARCH

        # ======================================================
        # 3. TEAM + FIXTURES ⇒ FIXTURE_QUERY (more specific)
        # ======================================================
        if (has_team_name or f["has_team_word"] or f["has_team_domain"]) and f["has_fixture_word"]:
            return Intent.FIXTURE_QUERY

        # ======================================================
        # 4. TEAM INTENTS (no explicit fixtures)
        # ======================================================
        if has_team_name or f["has_team_word"] or f["has_team_domain"]:
            return Intent.TEAM_QUERY

        # ======================================================
        # 5. GAMEWEEK / FIXTURES (without team / player)
        # ======================================================

        # 5a. Pure gameweek intents (no specific team / player)
        #     e.g. "What games are in gameweek 5?",
        #          "How many total gameweeks are there?"
        if (
            has_gameweek_context
            and not has_team_name
            and not f["has_team_word"]
            and not has_player_name
            and not f["has_player_word"]
        ):
            return Intent.GAMEWEEK_QUERY

        # 5b. Pure fixture intents (no gameweek context)
        #     e.g. "List all fixtures"
        if f["has_fixture_word"] and not has_gameweek_context:
            return Intent.FIXTURE_QUERY

        # ======================================================
        # 6. SEASON / GAMEWEEK GENERIC (no players/teams)
        # ======================================================
        # Has actual FPL statistics mentioned?
        has_fpl_stats = any(word in q_lower for word in [
            "goal", "goals", "point", "points", "assist", "assists", 
            "score", "scored", "clean sheet", "clean sheets"
        ])

        # Check for stat-related words (in case has_stat_word misses some)
        # has_any_stat_word = f["has_stat_word"] or any(word in q_lower for word in ["stat", "stats", "statistics"])

        # # Statistics queries with season context
        # if has_season_entity or f["has_season_word"] or f["has_season_domain"]:
        #     if has_any_stat_word or (f["has_aggregate_word"] and has_fpl_stats):
        #         return Intent.STATISTICS_QUERY
        
        
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
    print("=" * 60)
    print("FPL Intent Classification Test")
    print("=" * 60)
    queries = test_queries_5+test_queries_4+test_queries_3+test_queries_2+test_queries;
    for q in queries:
        intent, meta = classifier.classify(q)
        # hints = classifier.get_cypher_hints(intent)
        print(f"\nQuery: {q}")
        print(f"Intent: {intent.value}")
        print(f"Entities: {meta['entities']}")
        # print(f"Pattern:\n{hints['query_pattern']}")
        print("-" * 60)
# Query: How many total gameweeks are there?
# Intent: gameweek_query
# Entities: {'players': [], 'teams': [], 'seasons': [], 'gameweeks': [], 'fixtures': [], 'positions': [], 'stats': [], 'stat_ops': ['sum']}
# ------------------------------------------------------------

# Query: What is the highest points scored by a player?
# Intent: player_recommendation
# Entities: {'players': [], 'teams': [], 'seasons': [], 'gameweeks': [], 'fixtures': [], 'positions': [], 'stats': ['points', 'point'], 'stat_ops': ['max']}

# Query: How many total gameweeks are there?
# Intent: gameweek_query
# Entities: {'players': [], 'teams': [], 'seasons': [], 'gameweeks': [], 'fixtures': [], 'positions': [], 'stats': [], 'stat_ops': ['sum']}

# Query: What is the highest points scored by a player?
# Intent: player_recommendation
# Entities: {'players': [], 'teams': [], 'seasons': [], 'gameweeks': [], 'fixtures': [], 'positions': [], 'stats': ['points', 'point'], 'stat_ops': ['max']}