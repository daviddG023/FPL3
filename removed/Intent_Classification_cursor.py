"""
Intent Classification System for FPL (Fantasy Premier League) Queries

This module classifies user queries to determine the appropriate retrieval strategy
and Cypher query patterns for the FPL knowledge graph.

Intents:
- PLAYER_PERFORMANCE: Query about specific player statistics/performance
- PLAYER_RECOMMENDATION: Get recommendations for top players, best picks
- TEAM_QUERY: Questions about teams
- FIXTURE_QUERY: Questions about fixtures/matches
- GAMEWEEK_QUERY: Questions about gameweeks
- SEASON_QUERY: Questions about seasons
- STATISTICS_QUERY: General statistics queries
- COMPARISON_QUERY: Comparing players/teams
- ENTITY_SEARCH: Searching for specific entities
- POSITION_QUERY: Questions about player positions
"""

import re
from typing import Dict, List, Tuple, Optional
from enum import Enum


class Intent(Enum):
    """Intent categories for FPL queries"""
    PLAYER_PERFORMANCE = "player_performance"
    PLAYER_RECOMMENDATION = "player_recommendation"
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
    Classifies user queries into intent categories for FPL domain.
    
    Supports both rule-based (keyword matching) and LLM-based classification.
    """
    
    def __init__(self, use_llm: bool = False, llm_model: Optional[str] = None):
        """
        Initialize the intent classifier.
        
        Args:
            use_llm: Whether to use LLM-based classification (default: False, uses rule-based)
            llm_model: LLM model name if using LLM-based classification
        """
        self.use_llm = use_llm
        self.llm_model = llm_model
        
        # Define keyword patterns for each intent
        self.intent_patterns = self._build_intent_patterns()
        
        # FPL-specific entity keywords
        self.player_keywords = [
            'player', 'players', 'scorer', 'scorers', 'forward', 'midfielder',
            'defender', 'goalkeeper', 'gk', 'def', 'mid', 'fwd'
        ]
        
        self.team_keywords = [
            'team', 'teams', 'club', 'clubs', 'arsenal', 'chelsea', 'liverpool',
            'manchester', 'city', 'united', 'tottenham', 'spurs', 'leicester',
            'brighton', 'crystal palace', 'everton', 'fulham', 'leeds', 'newcastle',
            'southampton', 'west ham', 'wolves', 'aston villa', 'brentford',
            'burnley', 'norwich', 'watford', 'sheffield', 'bournemouth'
        ]
        
        self.fixture_keywords = [
            'fixture', 'fixtures', 'match', 'matches', 'game', 'games',
            'kickoff', 'kickoff time', 'vs', 'versus', 'against'
        ]
        
        self.gameweek_keywords = [
            'gameweek', 'game week', 'gw', 'week', 'round'
        ]
        
        self.season_keywords = [
            'season', 'seasons', '2021-22', '2022-23', '2023-24', '2024-25'
        ]
        
        self.position_keywords = [
            'position', 'positions', 'gk', 'goalkeeper', 'def', 'defender',
            'mid', 'midfielder', 'fwd', 'forward', 'attacker'
        ]
        
        self.stat_keywords = [
            'points', 'goals', 'assists', 'clean sheet', 'clean sheets',
            'yellow card', 'red card', 'cards', 'saves', 'bonus', 'bps',
            'influence', 'creativity', 'threat', 'ict', 'ict index', 'form',
            'minutes', 'played', 'appearances'
        ]
        
        self.recommendation_keywords = [
            'top', 'best', 'highest', 'most', 'recommend', 'recommendation',
            'pick', 'picks', 'should i', 'who should', 'which player',
            'suggest', 'advice'
        ]
        
        self.comparison_keywords = [
            'compare', 'comparison', 'vs', 'versus', 'better', 'difference',
            'between', 'than', 'versus'
        ]
    
    def _build_intent_patterns(self) -> Dict[Intent, List[str]]:
        """Build keyword patterns for each intent category"""
        return {
            Intent.PLAYER_PERFORMANCE: [
                r'\b(how many|what|show|display|get|find)\b.*\b(player|scorer|forward|midfielder|defender|goalkeeper)\b',
                r'\b(player|scorer)\b.*\b(points|goals|assists|stats|statistics|performance|played|minutes)\b',
                r'\b(how many|how much|what)\b.*\b(points|goals|assists|clean sheets|cards)\b.*\b(player|scorer)\b',
                r'\b(player|scorer)\b.*\b(name|named)\b.*\b(scored|got|has|have)\b',
            ],
            
            Intent.PLAYER_RECOMMENDATION: [
                r'\b(top|best|highest|most)\b.*\b(player|players|scorer|scorers|pick|picks)\b',
                r'\b(who|which)\b.*\b(best|top|highest|most|recommended)\b.*\b(player|players)\b',
                r'\b(should i|recommend|suggest|advice|pick|choose)\b.*\b(player|players)\b',
                r'\b(player|players)\b.*\b(with|having)\b.*\b(most|highest|best)\b',
                r'\b(best|top)\b.*\b(for|in)\b.*\b(position|gk|def|mid|fwd)\b',
            ],
            
            Intent.TEAM_QUERY: [
                r'\b(team|teams|club|clubs)\b',
                r'\b(which|what|how many)\b.*\b(team|teams)\b',
                r'\b(team|teams)\b.*\b(played|play|fixture|match|game)\b',
                r'\b(arsenal|chelsea|liverpool|manchester|city|united|tottenham|spurs)\b',
            ],
            
            Intent.FIXTURE_QUERY: [
                r'\b(fixture|fixtures|match|matches|game|games)\b',
                r'\b(when|what time|kickoff|kickoff time)\b.*\b(fixture|match|game)\b',
                r'\b(team|teams)\b.*\b(vs|versus|against)\b',
                r'\b(fixture|match|game)\b.*\b(number|#)\b',
            ],
            
            Intent.GAMEWEEK_QUERY: [
                r'\b(gameweek|game week|gw)\b',
                r'\b(week|round)\b.*\b(number|#)\b',
                r'\b(how many|what)\b.*\b(gameweek|game week|gw)\b',
                r'\b(fixture|match|game)\b.*\b(gameweek|game week|gw)\b',
            ],
            
            Intent.SEASON_QUERY: [
                r'\b(season|seasons)\b',
                r'\b(2021-22|2022-23|2023-24|2024-25)\b',
                r'\b(which|what|how many)\b.*\b(season|seasons)\b',
            ],
            
            Intent.STATISTICS_QUERY: [
                r'\b(count|total|how many|statistics|stats)\b',
                r'\b(aggregate|sum|average|avg|mean)\b',
                r'\b(most|least|highest|lowest)\b.*\b(goals|assists|points|clean sheets)\b',
            ],
            
            Intent.COMPARISON_QUERY: [
                r'\b(compare|comparison)\b',
                r'\b(vs|versus|against)\b',
                r'\b(better|worse|more|less|than)\b.*\b(than|vs|versus)\b',
                r'\b(difference|different|between)\b',
            ],
            
            Intent.POSITION_QUERY: [
                r'\b(position|positions)\b',
                r'\b(gk|goalkeeper|def|defender|mid|midfielder|fwd|forward)\b',
                r'\b(which|what)\b.*\b(position|positions)\b',
                r'\b(player|players)\b.*\b(position|positions)\b',
            ],
            
            Intent.ENTITY_SEARCH: [
                r'\b(find|search|look for|show|list|get)\b.*\b(player|team|fixture|gameweek|season)\b',
                r'\b(who|what|which|where)\b.*\b(is|are|was|were)\b',
            ],
        }
    
    def classify(self, query: str) -> Tuple[Intent, float, Dict[str, any]]:
        """
        Classify a user query into an intent category.
        
        Args:
            query: User query string
            
        Returns:
            Tuple of (intent, confidence_score, metadata)
            - intent: The classified Intent enum
            - confidence_score: Confidence score between 0 and 1
            - metadata: Additional information (extracted entities, keywords, etc.)
        """
        original_query = query.strip()
        query_lower = original_query.lower()
        
        if self.use_llm:
            return self._classify_with_llm(query_lower, original_query)
        else:
            return self._classify_with_rules(query_lower, original_query)
    
    def _classify_with_rules(self, query_lower: str, original_query: str) -> Tuple[Intent, float, Dict[str, any]]:
        """
        Rule-based classification using keyword matching and pattern matching.
        
        Args:
            query_lower: Lowercased user query
            original_query: Original user query (preserves case)
            
        Returns:
            Tuple of (intent, confidence_score, metadata)
        """
        intent_scores = {}
        metadata = {
            'matched_keywords': [],
            'matched_patterns': [],
            'entities': self._extract_entities(original_query)
        }
        
        # Score each intent based on pattern matches
        for intent, patterns in self.intent_patterns.items():
            score = 0.0
            matched_patterns = []
            
            for pattern in patterns:
                matches = re.findall(pattern, query_lower, re.IGNORECASE)
                if matches:
                    score += len(matches) * 0.3
                    matched_patterns.append(pattern)
            
            # Additional scoring based on keyword presence
            keyword_score = self._calculate_keyword_score(query_lower, intent)
            score += keyword_score
            
            if score > 0:
                intent_scores[intent] = score
                if matched_patterns:
                    metadata['matched_patterns'].extend(matched_patterns)
        
        # Handle special cases and boost scores (use original query for pattern matching)
        intent_scores = self._apply_special_rules(original_query, intent_scores, metadata)
        
        # Select best intent
        if not intent_scores:
            return Intent.UNKNOWN, 0.0, metadata
        
        best_intent = max(intent_scores.items(), key=lambda x: x[1])
        max_score = best_intent[1]
        
        # Normalize confidence score (cap at 1.0)
        confidence = min(max_score / 2.0, 1.0) if max_score > 0 else 0.0
        
        return best_intent[0], confidence, metadata
    
    def _calculate_keyword_score(self, query: str, intent: Intent) -> float:
        """Calculate score based on keyword presence"""
        score = 0.0
        
        if intent == Intent.PLAYER_PERFORMANCE:
            score += sum(0.2 for kw in self.player_keywords if kw in query)
            score += sum(0.15 for kw in self.stat_keywords if kw in query)
        
        elif intent == Intent.PLAYER_RECOMMENDATION:
            score += sum(0.25 for kw in self.recommendation_keywords if kw in query)
            score += sum(0.15 for kw in self.player_keywords if kw in query)
        
        elif intent == Intent.TEAM_QUERY:
            score += sum(0.2 for kw in self.team_keywords if kw in query)
        
        elif intent == Intent.FIXTURE_QUERY:
            score += sum(0.2 for kw in self.fixture_keywords if kw in query)
        
        elif intent == Intent.GAMEWEEK_QUERY:
            score += sum(0.25 for kw in self.gameweek_keywords if kw in query)
        
        elif intent == Intent.SEASON_QUERY:
            score += sum(0.25 for kw in self.season_keywords if kw in query)
        
        elif intent == Intent.STATISTICS_QUERY:
            score += sum(0.2 for kw in self.stat_keywords if kw in query)
            if any(kw in query for kw in ['count', 'total', 'how many', 'sum', 'average']):
                score += 0.3
        
        elif intent == Intent.COMPARISON_QUERY:
            score += sum(0.3 for kw in self.comparison_keywords if kw in query)
        
        elif intent == Intent.POSITION_QUERY:
            score += sum(0.25 for kw in self.position_keywords if kw in query)
        
        elif intent == Intent.ENTITY_SEARCH:
            if any(kw in query for kw in ['find', 'search', 'show', 'list', 'get']):
                score += 0.3
        
        return score
    
    def _apply_special_rules(self, query: str, intent_scores: Dict[Intent, float], 
                            metadata: Dict) -> Dict[Intent, float]:
        """Apply special rules to boost or adjust intent scores"""
        
        # If query contains player name pattern, boost PLAYER_PERFORMANCE
        # Check both original case and metadata entities
        player_names = metadata.get('entities', {}).get('players', [])
        if player_names:
            # Filter out false positives (like "Compare Mohamed" when we want "Mohamed Salah")
            valid_player_names = [name for name in player_names if len(name.split()) >= 2 and 
                                 not any(word.lower() in ['compare', 'show', 'find', 'get', 'list'] for word in name.split()[:2])]
            if valid_player_names:
                # Strong boost for queries about specific players
                intent_scores[Intent.PLAYER_PERFORMANCE] = intent_scores.get(
                    Intent.PLAYER_PERFORMANCE, 0) + 0.7
                # If asking "how many/what" about a specific player, it's definitely PLAYER_PERFORMANCE
                if re.search(r'\b(how many|what|how much|show|get|find)\b', query, re.IGNORECASE):
                    intent_scores[Intent.PLAYER_PERFORMANCE] = intent_scores.get(
                        Intent.PLAYER_PERFORMANCE, 0) + 0.5
                    # Reduce STATISTICS_QUERY score if it was boosted
                    if Intent.STATISTICS_QUERY in intent_scores:
                        intent_scores[Intent.STATISTICS_QUERY] *= 0.5
        
        # If query asks to "find/show/list players" without specific stats, it's ENTITY_SEARCH
        if re.search(r'\b(find|show|list|get|display)\b.*\b(player|players)\b', query, re.IGNORECASE):
            if not any(kw in query for kw in ['points', 'goals', 'assists', 'stats', 'performance']):
                intent_scores[Intent.ENTITY_SEARCH] = intent_scores.get(
                    Intent.ENTITY_SEARCH, 0) + 0.5
                # Reduce PLAYER_PERFORMANCE score if it was boosted
                if Intent.PLAYER_PERFORMANCE in intent_scores:
                    intent_scores[Intent.PLAYER_PERFORMANCE] *= 0.5
        
        # If query asks about "players who play as" or "position", boost POSITION_QUERY
        if re.search(r'\b(player|players)\b.*\b(who|that)\b.*\b(play|plays)\b.*\b(as|position)\b', query, re.IGNORECASE):
            intent_scores[Intent.POSITION_QUERY] = intent_scores.get(
                Intent.POSITION_QUERY, 0) + 0.6
            # Reduce PLAYER_PERFORMANCE score
            if Intent.PLAYER_PERFORMANCE in intent_scores:
                intent_scores[Intent.PLAYER_PERFORMANCE] *= 0.4
        
        # If query asks "who" or "which player", likely PLAYER_RECOMMENDATION or ENTITY_SEARCH
        if re.search(r'\b(who|which player)\b', query, re.IGNORECASE):
            if any(kw in query for kw in ['top', 'best', 'most', 'highest']):
                intent_scores[Intent.PLAYER_RECOMMENDATION] = intent_scores.get(
                    Intent.PLAYER_RECOMMENDATION, 0) + 0.4
            else:
                intent_scores[Intent.ENTITY_SEARCH] = intent_scores.get(
                    Intent.ENTITY_SEARCH, 0) + 0.3
        
        # If query contains "how many", likely STATISTICS_QUERY
        if 'how many' in query:
            intent_scores[Intent.STATISTICS_QUERY] = intent_scores.get(
                Intent.STATISTICS_QUERY, 0) + 0.3
        
        # If query contains comparison keywords with player/team mentions
        if any(kw in query for kw in ['vs', 'versus', 'better', 'than']):
            if any(kw in query for kw in self.player_keywords + self.team_keywords):
                intent_scores[Intent.COMPARISON_QUERY] = intent_scores.get(
                    Intent.COMPARISON_QUERY, 0) + 0.4
        
        return intent_scores
    
    def _extract_entities(self, query: str) -> Dict[str, List[str]]:
        """Extract entities from the query"""
        entities = {
            'players': [],
            'teams': [],
            'seasons': [],
            'gameweeks': [],
            'fixtures': [],
            'positions': [],
            'stats': []
        }
        
        # Extract player names (capitalized words, typically "First Last")
        # Pattern: Capitalized word followed by another capitalized word
        player_name_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b'
        player_matches = re.findall(player_name_pattern, query)
        
        # Filter out common false positives (team names, common words)
        false_positives = {'Manchester United', 'Manchester City', 'Crystal Palace', 
                          'West Ham', 'Aston Villa', 'Sheffield United', 'Brighton Hove'}
        filtered_matches = [name for name in player_matches if name not in false_positives]
        
        # Also extract common FPL player name patterns
        # Handle names like "Mohamed Salah", "Erling Haaland", etc.
        # Look for sequences of capitalized words (2-3 words typically)
        words = query.split()
        for i in range(len(words) - 1):
            # Check if we have two consecutive capitalized words
            if (len(words[i]) > 1 and words[i][0].isupper() and words[i][1:].islower() and
                len(words[i+1]) > 1 and words[i+1][0].isupper() and words[i+1][1:].islower()):
                potential_name = f"{words[i]} {words[i+1]}"
                # Avoid duplicates and common false positives
                if (potential_name not in filtered_matches and 
                    potential_name not in false_positives and
                    words[i].lower() not in ['compare', 'show', 'find', 'get', 'list', 'which', 'what', 'who']):
                    filtered_matches.append(potential_name)
        
        # Remove duplicates while preserving order
        # Also filter out names that start with common query words
        seen = set()
        entities['players'] = []
        query_start_words = {'compare', 'show', 'find', 'get', 'list', 'which', 'what', 'who', 'when', 'where'}
        
        for name in filtered_matches:
            # Skip if name starts with a query word
            first_word = name.split()[0].lower() if name.split() else ''
            if first_word in query_start_words:
                continue
            if name not in seen:
                seen.add(name)
                entities['players'].append(name)
        
        # Extract team names
        for team in self.team_keywords:
            if team.lower() in query.lower():
                entities['teams'].append(team)
        
        # Extract seasons
        season_pattern = r'\b(202[1-9]-[0-9]{2})\b'
        entities['seasons'] = re.findall(season_pattern, query)
        
        # Extract gameweek numbers
        gw_pattern = r'\b(gw|gameweek|game week)\s*(\d+)\b'
        gw_matches = re.findall(gw_pattern, query, re.IGNORECASE)
        entities['gameweeks'] = [match[1] for match in gw_matches]
        
        # Extract fixture numbers
        fixture_pattern = r'\b(fixture|match|game)\s*(?:number|#)?\s*(\d+)\b'
        fixture_matches = re.findall(fixture_pattern, query, re.IGNORECASE)
        entities['fixtures'] = [match[1] for match in fixture_matches]
        
        # Extract positions
        for pos in self.position_keywords:
            if pos in query:
                entities['positions'].append(pos)
        
        # Extract stats
        for stat in self.stat_keywords:
            if stat in query:
                entities['stats'].append(stat)
        
        return entities
    
    def _classify_with_llm(self, query_lower: str, original_query: str) -> Tuple[Intent, float, Dict[str, any]]:
        """
        LLM-based classification (placeholder for future implementation).
        
        Args:
            query_lower: Lowercased user query
            original_query: Original user query (preserves case)
            
        Returns:
            Tuple of (intent, confidence_score, metadata)
        """
        # TODO: Implement LLM-based classification
        # This would use an LLM to classify the intent more accurately
        # For now, fall back to rule-based
        return self._classify_with_rules(query_lower, original_query)
    
    def get_cypher_hints(self, intent: Intent) -> Dict[str, any]:
        """
        Get Cypher query hints based on the intent.
        
        Args:
            intent: The classified intent
            
        Returns:
            Dictionary with hints for constructing Cypher queries
        """
        hints = {
            Intent.PLAYER_PERFORMANCE: {
                'nodes': ['Player', 'Fixture'],
                'relationships': ['PLAYED_IN'],
                'properties': ['total_points', 'goals_scored', 'assists', 'minutes', 
                              'clean_sheets', 'bonus', 'bps', 'ict_index', 'form'],
                'query_pattern': 'MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture) WHERE ...'
            },
            
            Intent.PLAYER_RECOMMENDATION: {
                'nodes': ['Player', 'Fixture'],
                'relationships': ['PLAYED_IN'],
                'properties': ['total_points', 'goals_scored', 'assists', 'ict_index', 'form'],
                'query_pattern': 'MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture) ... ORDER BY r.total_points DESC LIMIT N',
                'aggregation': True
            },
            
            Intent.TEAM_QUERY: {
                'nodes': ['Team', 'Fixture'],
                'relationships': ['HAS_HOME_TEAM', 'HAS_AWAY_TEAM'],
                'properties': ['name'],
                'query_pattern': 'MATCH (f:Fixture)-[:HAS_HOME_TEAM|HAS_AWAY_TEAM]->(t:Team) WHERE ...'
            },
            
            Intent.FIXTURE_QUERY: {
                'nodes': ['Fixture', 'Team', 'Gameweek'],
                'relationships': ['HAS_FIXTURE', 'HAS_HOME_TEAM', 'HAS_AWAY_TEAM'],
                'properties': ['fixture_number', 'kickoff_time', 'season'],
                'query_pattern': 'MATCH (gw:Gameweek)-[:HAS_FIXTURE]->(f:Fixture) WHERE ...'
            },
            
            Intent.GAMEWEEK_QUERY: {
                'nodes': ['Gameweek', 'Season', 'Fixture'],
                'relationships': ['HAS_GW', 'HAS_FIXTURE'],
                'properties': ['GW_number', 'season'],
                'query_pattern': 'MATCH (s:Season)-[:HAS_GW]->(gw:Gameweek) WHERE ...'
            },
            
            Intent.SEASON_QUERY: {
                'nodes': ['Season', 'Gameweek'],
                'relationships': ['HAS_GW'],
                'properties': ['season_name'],
                'query_pattern': 'MATCH (s:Season) WHERE s.season_name = ...'
            },
            
            Intent.STATISTICS_QUERY: {
                'nodes': ['Player', 'Team', 'Fixture'],
                'relationships': ['PLAYED_IN'],
                'properties': ['total_points', 'goals_scored', 'assists'],
                'query_pattern': 'MATCH ... RETURN COUNT(...) or SUM(...) or AVG(...)',
                'aggregation': True
            },
            
            Intent.COMPARISON_QUERY: {
                'nodes': ['Player', 'Team', 'Fixture'],
                'relationships': ['PLAYED_IN'],
                'properties': ['total_points', 'goals_scored', 'assists'],
                'query_pattern': 'MATCH ... WITH ... WHERE ... comparison logic'
            },
            
            Intent.POSITION_QUERY: {
                'nodes': ['Player', 'Position'],
                'relationships': ['PLAYS_AS'],
                'properties': ['name'],
                'query_pattern': 'MATCH (p:Player)-[:PLAYS_AS]->(pos:Position) WHERE ...'
            },
            
            Intent.ENTITY_SEARCH: {
                'nodes': ['Player', 'Team', 'Fixture', 'Gameweek', 'Season'],
                'relationships': ['PLAYED_IN', 'HAS_HOME_TEAM', 'HAS_AWAY_TEAM', 'HAS_FIXTURE'],
                'properties': ['player_name', 'name', 'season_name'],
                'query_pattern': 'MATCH (entity:EntityType) WHERE entity.property CONTAINS ...'
            },
            
            Intent.UNKNOWN: {
                'nodes': [],
                'relationships': [],
                'properties': [],
                'query_pattern': 'Unable to determine query pattern'
            }
        }
        
        return hints.get(intent, hints[Intent.UNKNOWN])


# Example usage and testing
if __name__ == "__main__":
    classifier = IntentClassifier(use_llm=False)
    
    # Test queries
    test_queries = [
        "How many points did Mohamed Salah score?",
        "Who are the top 10 players with most goals?",
        "Show me fixtures for Manchester United",
        "What games are in gameweek 5?",
        "Which teams played in the 2022-23 season?",
        "Compare Mohamed Salah vs Erling Haaland",
        "Find players who play as defender",
        "How many total gameweeks are there?",
        "What is the highest points scored by a player?",
        "Show me all players",
    ]
    
    print("=" * 80)
    print("FPL Intent Classification Test")
    print("=" * 80)
    
    for query in test_queries:
        intent, confidence, metadata = classifier.classify(query)
        hints = classifier.get_cypher_hints(intent)
        
        print(f"\nQuery: {query}")
        print(f"Intent: {intent.value}")
        print(f"Confidence: {confidence:.2f}")
        print(f"Entities: {metadata['entities']}")
        print(f"Cypher Hints: {hints['query_pattern']}")
        print("-" * 80)

