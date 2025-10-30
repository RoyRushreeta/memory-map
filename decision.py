"""
🧭 DECISION MODULE - Reasoning and Decision Making

This module handles reasoning about what to do with the retrieved results.
For now, it decides to show the top results on the map, but this can be
extended for more complex decision-making logic.
"""

import pandas as pd
from typing import Dict, Any, List, Tuple


class Decision:
    """
    Handles reasoning about what actions to take based on retrieved results.
    
    This class is responsible for:
    - Analyzing retrieved memory results
    - Deciding what actions to take (e.g., show on map, display message)
    - Determining display parameters (how many to highlight, zoom level, etc.)
    """
    
    def __init__(self):
        """Initialize the Decision module."""
        self.default_top_n = 3
        self.min_similarity_threshold = 0.0  # Minimum similarity score to consider
    
    def decide_action(self, query: str, results: pd.DataFrame, scores: List[float]) -> Dict[str, Any]:
        """
        Decide what action to take based on the query and retrieved results.
        
        Args:
            query (str): The original user query
            results (pd.DataFrame): Retrieved memory results
            scores (List[float]): Similarity scores for the results
            
        Returns:
            Dict[str, Any]: Decision containing action type and parameters
        """
        if results.empty or (scores is None) or (len(scores) == 0):
            return self._decide_no_results(query)
        
        # Filter results based on similarity threshold
        filtered_results = self._filter_by_similarity(results, scores)
        
        if filtered_results.empty:
            return self._decide_low_similarity(query)
        
        return self._decide_show_results(query, filtered_results, scores)
    
    def _decide_no_results(self, query: str) -> Dict[str, Any]:
        """
        Decision when no results are found.
        
        Args:
            query (str): The original user query
            
        Returns:
            Dict[str, Any]: Decision for no results scenario
        """
        return {
            'action': 'show_all_memories',
            'message': f"No specific memories found for '{query}'. Showing all memories on the map.",
            'highlight_count': 0,
            'show_search_results': False,
            'zoom_to_results': False
        }
    
    def _decide_low_similarity(self, query: str) -> Dict[str, Any]:
        """
        Decision when similarity scores are too low.
        
        Args:
            query (str): The original user query
            
        Returns:
            Dict[str, Any]: Decision for low similarity scenario
        """
        return {
            'action': 'show_all_memories',
            'message': f"No closely matching memories found for '{query}'. Showing all memories on the map.",
            'highlight_count': 0,
            'show_search_results': False,
            'zoom_to_results': False
        }
    
    def _decide_show_results(self, query: str, results: pd.DataFrame, scores: List[float]) -> Dict[str, Any]:
        """
        Decision when good results are found.
        
        Args:
            query (str): The original user query
            results (pd.DataFrame): Filtered results
            scores (List[float]): Similarity scores
            
        Returns:
            Dict[str, Any]: Decision for showing results
        """
        top_n = min(self.default_top_n, len(results))
        
        return {
            'action': 'show_search_results',
            'message': f"Showing top {top_n} similar memories for: '{query}'",
            'highlight_count': top_n,
            'show_search_results': True,
            'zoom_to_results': True,
            'top_n': top_n
        }
    
    def _filter_by_similarity(self, results: pd.DataFrame, scores: List[float]) -> pd.DataFrame:
        """
        Filter results based on similarity threshold.
        
        Args:
            results (pd.DataFrame): Results to filter
            scores (List[float]): Similarity scores
            
        Returns:
            pd.DataFrame: Filtered results
        """
        if (scores is None) or (len(scores) == 0):
            return pd.DataFrame()
        
        # Add scores to results for filtering
        results_with_scores = results.copy()
        results_with_scores['similarity_score'] = scores[:len(results)]
        
        # Filter by threshold
        filtered = results_with_scores[results_with_scores['similarity_score'] >= self.min_similarity_threshold]
        
        return filtered.drop(columns=['similarity_score'])
    
    def determine_map_bounds(self, results: pd.DataFrame) -> List[List[float]]:
        """
        Determine the bounds for the map based on results.
        
        Args:
            results (pd.DataFrame): Results to calculate bounds for
            
        Returns:
            List[List[float]]: Bounds in format [[min_lat, min_lon], [max_lat, max_lon]]
        """
        if results.empty or 'latitude' not in results.columns or 'longitude' not in results.columns:
            return None
        
        min_lat = float(results['latitude'].min())
        max_lat = float(results['latitude'].max())
        min_lon = float(results['longitude'].min())
        max_lon = float(results['longitude'].max())
        
        return [[min_lat, min_lon], [max_lat, max_lon]]
    
    def should_show_image_grid(self, results: pd.DataFrame) -> bool:
        """
        Decide whether to show the image grid for search results.
        
        Args:
            results (pd.DataFrame): Search results
            
        Returns:
            bool: True if image grid should be shown
        """
        return not results.empty and len(results) > 0
    
    def get_display_count(self, results: pd.DataFrame) -> int:
        """
        Get the number of results to display in the image grid.
        
        Args:
            results (pd.DataFrame): Search results
            
        Returns:
            int: Number of results to display
        """
        return min(self.default_top_n, len(results))
    
    def analyze_query_intent(self, query: str) -> Dict[str, Any]:
        """
        Analyze the user's query to understand intent (can be extended).
        
        Args:
            query (str): User's query
            
        Returns:
            Dict[str, Any]: Analysis of query intent
        """
        query_lower = query.lower()
        
        # Simple keyword-based analysis (can be made more sophisticated)
        intent_keywords = {
            'location': ['where', 'place', 'location', 'city', 'country'],
            'activity': ['trip', 'vacation', 'visit', 'went', 'travel'],
            'emotion': ['happy', 'sad', 'beautiful', 'amazing', 'wonderful'],
            'time': ['recent', 'old', 'last', 'first', 'when'],
            'visual': ['photo', 'picture', 'click', 'image', 'shot']
        }
        
        detected_intents = []
        for intent, keywords in intent_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                detected_intents.append(intent)
        
        return {
            'query': query,
            'detected_intents': detected_intents,
            'has_specific_intent': len(detected_intents) > 0
        }
    
    def set_similarity_threshold(self, threshold: float):
        """
        Set the minimum similarity threshold for filtering results.
        
        Args:
            threshold (float): Minimum similarity score (0.0 to 1.0)
        """
        self.min_similarity_threshold = max(0.0, min(1.0, threshold))
    
    def set_default_top_n(self, top_n: int):
        """
        Set the default number of top results to highlight.
        
        Args:
            top_n (int): Number of top results to highlight
        """
        self.default_top_n = max(1, top_n)