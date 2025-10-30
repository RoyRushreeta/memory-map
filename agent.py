"""
🧠 AGENT MODULE - Orchestrator and Coordinator

This is the main orchestrator file that imports and coordinates the other 4 modules 
(Perception, Memory, Decision, Action). It provides a unified interface for processing 
user queries and generating responses.
"""

import pandas as pd
import folium
from typing import Dict, Any, Tuple, Optional

from perception import Perception
from memory import Memory
from decision import Decision
from action import Action


class Agent:
    """
    Main orchestrator class that coordinates all modules to process user queries.
    
    This class is responsible for:
    - Coordinating the Perception, Memory, Decision, and Action modules
    - Processing user queries through the complete pipeline
    - Managing the flow of information between modules
    """
    
    def __init__(self, csv_path="memories.csv"):
        """
        Initialize the Agent with all necessary modules.
        
        Args:
            csv_path (str): Path to the CSV file containing memory data
        """
        # Initialize all modules
        self.perception = Perception()
        self.memory = Memory(csv_path)
        self.decision = Decision()
        self.action = Action()
        
        # Verify that memory data is loaded successfully
        if not self.memory.is_data_loaded():
            raise RuntimeError("Failed to load memory data. Please check your CSV file.")
    
    def respond_to_query(self, query: str) -> Tuple[folium.Map, Dict[str, Any]]:
        """
        Main method to process a user query and return the appropriate response.
        
        This method orchestrates the complete pipeline:
        1. Uses Perception to process the user's query into embeddings
        2. Uses Memory to retrieve the top-k similar memories
        3. Uses Decision to decide the next step (e.g., show results)
        4. Uses Action to display the results (highlight top 3 on map)
        
        Args:
            query (str): User's natural language query
            
        Returns:
            Tuple[folium.Map, Dict[str, Any]]: (map_object, response_info)
        """
        # Step 1: Use Perception to process the query
        query_embedding = self.perception.encode_query(query)
        
        # Step 2: Use Memory to retrieve similar memories
        if query_embedding is not None:
            scores, indices, matched_df = self.memory.search_similar_memories(query_embedding, k=10)
            # Limit to top 3 for display consistency with original app
            matched_df = matched_df.head(3) if not matched_df.empty else matched_df
            scores = scores[:3] if scores is not None else []
        else:
            matched_df = pd.DataFrame()
            scores = []
        
        # Step 3: Use Decision to decide what to do
        decision = self.decision.decide_action(query, matched_df, scores)
        
        # Step 4: Use Action to execute the decision
        all_memories = self.memory.get_all_memories()
        map_obj = self.action.execute_action(decision, matched_df, all_memories)
        
        # Prepare response information
        response_info = {
            'query': query,
            'decision': decision,
            'results_count': len(matched_df),
            'total_memories': len(all_memories),
            'has_results': not matched_df.empty
        }
        
        return map_obj, response_info
    
    def get_all_memories_map(self) -> folium.Map:
        """
        Get a map showing all memories without any query filtering.
        
        Returns:
            folium.Map: Map with all memories displayed
        """
        all_memories = self.memory.get_all_memories()
        return self.action.create_all_memories_map(all_memories)
    
    def search_memories(self, query: str, k: int = 10) -> Tuple[pd.DataFrame, list]:
        """
        Search for memories similar to the query without executing actions.
        
        Args:
            query (str): Search query
            k (int): Number of results to retrieve
            
        Returns:
            Tuple[pd.DataFrame, list]: (results_dataframe, similarity_scores)
        """
        query_embedding = self.perception.encode_query(query)
        
        if query_embedding is not None:
            scores, indices, matched_df = self.memory.search_similar_memories(query_embedding, k=k)
            return matched_df, scores if scores is not None else []
        else:
            return pd.DataFrame(), []
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the loaded memories and system status.
        
        Returns:
            Dict[str, Any]: System and data statistics
        """
        memory_info = self.memory.get_data_info()
        perception_info = self.perception.get_model_info()
        
        return {
            'memory_stats': memory_info,
            'perception_stats': perception_info,
            'system_ready': self.memory.is_data_loaded(),
            'modules_initialized': {
                'perception': self.perception is not None,
                'memory': self.memory is not None,
                'decision': self.decision is not None,
                'action': self.action is not None
            }
        }
    
    def configure_system(self, config: Dict[str, Any]):
        """
        Configure system parameters.
        
        Args:
            config (Dict[str, Any]): Configuration parameters
        """
        # Configure decision module
        if 'similarity_threshold' in config:
            self.decision.set_similarity_threshold(config['similarity_threshold'])
        
        if 'default_top_n' in config:
            self.decision.set_default_top_n(config['default_top_n'])
        
        # Configure action module
        if 'image_settings' in config:
            image_config = config['image_settings']
            self.action.set_image_settings(
                max_width=image_config.get('max_width', 400),
                max_height=image_config.get('max_height', 300),
                quality=image_config.get('quality', 85)
            )
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Analyze a query to understand its intent and characteristics.
        
        Args:
            query (str): Query to analyze
            
        Returns:
            Dict[str, Any]: Query analysis results
        """
        return self.decision.analyze_query_intent(query)
    
    def get_memory_by_location(self, location: str) -> pd.DataFrame:
        """
        Get memories that match a specific location.
        
        Args:
            location (str): Location name to search for
            
        Returns:
            pd.DataFrame: Memories matching the location
        """
        all_memories = self.memory.get_all_memories()
        
        if 'location' not in all_memories.columns:
            return pd.DataFrame()
        
        # Case-insensitive partial matching
        location_matches = all_memories[
            all_memories['location'].str.contains(location, case=False, na=False)
        ]
        
        return location_matches
    
    def is_ready(self) -> bool:
        """
        Check if the agent is ready to process queries.
        
        Returns:
            bool: True if ready, False otherwise
        """
        return (
            self.perception is not None and
            self.memory is not None and
            self.memory.is_data_loaded() and
            self.decision is not None and
            self.action is not None
        )
    
    def get_system_info(self) -> str:
        """
        Get a formatted string with system information.
        
        Returns:
            str: System information string
        """
        if not self.is_ready():
            return "❌ System not ready - some modules failed to initialize"
        
        stats = self.get_memory_stats()
        memory_stats = stats['memory_stats']
        perception_stats = stats['perception_stats']
        
        info = f"""
        ✅ Memory Map Agent Ready
        
        📊 Memory Statistics:
        - Total memories: {memory_stats['total_memories']}
        - Embedding dimension: {memory_stats['embedding_dimension']}
        
        🧠 Perception Model:
        - Model: {perception_stats['model_name']}
        - Max sequence length: {perception_stats['max_seq_length']}
        
        🎯 All modules initialized and ready to serve queries!
        """
        
        return info.strip()