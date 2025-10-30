"""
💾 MEMORY MODULE - Data Storage and Retrieval

This module handles loading of memory data (captions, image paths, latitude, longitude, etc.),
precomputes or loads stored embeddings for each memory, and contains methods to compute 
similarity with query embeddings and retrieve top results.
"""

import pandas as pd
import faiss
import numpy as np
import streamlit as st
from perception import Perception


class Memory:
    """
    Handles memory data storage, embedding computation, and similarity search.
    
    This class is responsible for:
    - Loading memory data from CSV
    - Creating and managing FAISS index for similarity search
    - Retrieving top similar memories based on query embeddings
    """
    
    def __init__(self, csv_path="memories.csv"):
        """
        Initialize the Memory module.
        
        Args:
            csv_path (str): Path to the CSV file containing memory data
        """
        self.csv_path = csv_path
        self.df = self._load_data()
        self.perception = Perception()
        self.index, self.embeddings = self._create_faiss_index()
    
    @st.cache_data
    def _load_data(_self):
        """
        Load memory data from CSV file with caching.
        
        Returns:
            pd.DataFrame: DataFrame containing memory data
        """
        try:
            df = pd.read_csv(_self.csv_path)
            return df
        except Exception as e:
            st.error(f"Error loading CSV: {e}")
            return pd.DataFrame()
    
    @st.cache_resource
    def _create_faiss_index(_self):
        """
        Create FAISS index from text embeddings with caching.
        
        Returns:
            tuple: (FAISS index, embeddings array)
        """
        if _self.df.empty:
            return None, np.array([])
        
        # Combine location and caption text fields
        texts = (_self.df["location"].fillna('') + " " + _self.df["caption"].fillna('')).tolist()
        
        # Generate embeddings using the perception module
        embeddings = _self.perception.encode_texts(texts, show_progress=True)
        
        # Create FAISS index for cosine similarity (Inner Product on normalized vectors)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        
        return index, embeddings
    
    def search_similar_memories(self, query_embedding, k=10):
        """
        Search for similar memories based on query embedding.
        
        Args:
            query_embedding (np.ndarray): Query embedding vector
            k (int): Number of top similar memories to retrieve
            
        Returns:
            tuple: (similarity_scores, memory_indices, matched_dataframe)
        """
        if self.index is None or query_embedding is None:
            return None, None, pd.DataFrame()
        
        # Search for top k similar memories
        scores, indices = self.index.search(query_embedding, k=k)
        
        # Get the corresponding memory data
        matched_df = self.df.iloc[indices[0]].copy()
        matched_df['similarity_score'] = scores[0]
        
        return scores[0], indices[0], matched_df
    
    def get_memory_by_index(self, index):
        """
        Get a specific memory by its index.
        
        Args:
            index (int): Index of the memory in the dataframe
            
        Returns:
            pd.Series: Memory data for the specified index
        """
        if index < 0 or index >= len(self.df):
            return None
        return self.df.iloc[index]
    
    def get_all_memories(self):
        """
        Get all memories in the dataset.
        
        Returns:
            pd.DataFrame: Complete memory dataset
        """
        return self.df.copy()
    
    def get_memory_count(self):
        """
        Get the total number of memories in the dataset.
        
        Returns:
            int: Number of memories
        """
        return len(self.df)
    
    def get_text_for_memory(self, index):
        """
        Get the combined text (location + caption) for a specific memory.
        
        Args:
            index (int): Index of the memory
            
        Returns:
            str: Combined text for the memory
        """
        if index < 0 or index >= len(self.df):
            return ""
        
        row = self.df.iloc[index]
        location = str(row.get("location", "")).strip()
        caption = str(row.get("caption", "")).strip()
        return f"{location} {caption}".strip()
    
    def is_data_loaded(self):
        """
        Check if memory data has been successfully loaded.
        
        Returns:
            bool: True if data is loaded, False otherwise
        """
        return not self.df.empty
    
    def get_data_info(self):
        """
        Get information about the loaded data.
        
        Returns:
            dict: Data information including shape, columns, etc.
        """
        return {
            'total_memories': len(self.df),
            'columns': list(self.df.columns),
            'has_location': 'location' in self.df.columns,
            'has_caption': 'caption' in self.df.columns,
            'has_coordinates': 'latitude' in self.df.columns and 'longitude' in self.df.columns,
            'has_images': 'image' in self.df.columns,
            'embedding_dimension': self.embeddings.shape[1] if len(self.embeddings) > 0 else 0
        }