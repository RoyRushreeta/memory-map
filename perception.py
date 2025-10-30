"""
👁️ PERCEPTION MODULE - Input Understanding and Query Processing

This module handles all input understanding, specifically embedding text queries
using the SentenceTransformer model. It processes user queries into embeddings
that can be used for similarity search.
"""

import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np


class Perception:
    """
    Handles query understanding and text embedding.
    
    This class is responsible for:
    - Loading the embedding model
    - Converting user queries into embeddings
    - Normalizing embeddings for cosine similarity
    """
    
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initialize the Perception module.
        
        Args:
            model_name (str): Name of the SentenceTransformer model to use
        """
        self.model_name = model_name
        self.model = self._load_model()
    
    @st.cache_resource
    def _load_model(_self):
        """
        Load the SentenceTransformer model with caching.
        
        Returns:
            SentenceTransformer: The loaded embedding model
        """
        return SentenceTransformer(_self.model_name)
    
    def encode_query(self, query):
        """
        Convert a text query into a normalized embedding vector.
        
        Args:
            query (str): The user's text query
            
        Returns:
            np.ndarray: Normalized embedding vector for the query
        """
        if not query or not query.strip():
            return None
            
        # Encode the query
        query_embedding = self.model.encode([query])
        
        # Normalize for cosine similarity
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        return query_embedding.astype('float32')
    
    def encode_texts(self, texts, show_progress=True):
        """
        Convert a list of texts into normalized embedding vectors.
        
        Args:
            texts (list): List of text strings to embed
            show_progress (bool): Whether to show progress bar
            
        Returns:
            np.ndarray: Normalized embedding vectors for all texts
        """
        if not texts:
            return np.array([])
            
        # Encode all texts
        embeddings = self.model.encode(texts, show_progress_bar=show_progress)
        
        # Normalize for cosine similarity
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        return embeddings.astype('float32')
    
    def get_model_info(self):
        """
        Get information about the loaded model.
        
        Returns:
            dict: Model information
        """
        return {
            'model_name': self.model_name,
            'max_seq_length': getattr(self.model, 'max_seq_length', 'Unknown'),
            'embedding_dimension': self.model.get_sentence_embedding_dimension()
        }