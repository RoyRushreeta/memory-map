"""
⚡ ACTION MODULE - Visual Display and Execution

This module executes the visual display logic using Folium to plot results on the map,
show images and captions in popups, and highlight the top results in red while others in blue.
"""

import streamlit as st
import pandas as pd
import folium
import base64
import os
from PIL import Image
import io
from typing import Dict, Any, List


class Action:
    """
    Handles visual display and execution of actions.
    
    This class is responsible for:
    - Creating and managing Folium maps
    - Adding markers with popups to maps
    - Highlighting top results in different colors
    - Creating image grids for search results
    - Managing image processing and display
    """
    
    def __init__(self):
        """Initialize the Action module."""
        self.default_location = [20.5937, 78.9629]  # Center of India
        self.default_zoom = 5
        self.image_max_width = 400
        self.image_max_height = 300
        self.image_quality = 85
    
    @st.cache_data
    def resize_and_encode_image(_self, image_path, max_width=400, max_height=300, quality=85):
        """
        Resize image and convert to base64 string with caching.
        
        Args:
            image_path (str): Path to the image file
            max_width (int): Maximum width for resizing
            max_height (int): Maximum height for resizing
            quality (int): JPEG quality (1-100)
            
        Returns:
            str or None: Base64 encoded image string or None if failed
        """
        try:
            if not os.path.exists(image_path):
                return None

            with Image.open(image_path) as img:
                if img.mode in ('RGBA', 'LA', 'P'):
                    img = img.convert('RGB')
                img.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
                img_buffer = io.BytesIO()
                img.save(img_buffer, format='JPEG', quality=quality, optimize=True)
                img_buffer.seek(0)
                return base64.b64encode(img_buffer.getvalue()).decode()
        except Exception:
            return None
    
    def create_popup_html(self, row):
        """
        Create the HTML popup content for each marker.
        
        Args:
            row (pd.Series): Row data containing location, caption, and image information
            
        Returns:
            str: HTML content for the popup
        """
        location = str(row.get("location", "Unknown Location")).strip()
        caption = str(row.get("caption", "No description available")).strip()
        image_name = str(row.get("image", "")).strip()

        image_path = f"images/{image_name}"
        base64_image = self.resize_and_encode_image(
            image_path, 
            self.image_max_width, 
            self.image_max_height, 
            self.image_quality
        )

        if base64_image:
            html = f"""
            <div style="text-align:center; font-family: Arial, sans-serif;">
                <h4 style="margin: 5px 0; color: #2E86AB;"><b>{location}</b></h4>
                <p style="margin: 8px 0; font-size: 14px; color: #555;">{caption}</p>
                <img src="data:image/jpeg;base64,{base64_image}" 
                     style="width: 250px; max-height: 200px; object-fit: cover; border-radius: 8px; border: 2px solid #ddd;">
            </div>
            """
        else:
            html = f"""
            <div style="text-align:center; font-family: Arial, sans-serif;">
                <h4 style="margin: 5px 0; color: #2E86AB;"><b>{location}</b></h4>
                <p style="margin: 8px 0; font-size: 14px; color: #555;">{caption}</p>
                <div style="width: 250px; height: 150px; background-color: #f0f0f0;
                           border: 2px dashed #ccc; border-radius: 8px; display: flex;
                           align-items: center; justify-content: center; color: #999;">
                    <p>📷 Image not available</p>
                </div>
            </div>
            """
        return html
    
    @st.cache_data
    def create_base_map(_self, location=None, zoom_start=None):
        """
        Create a base Folium map with caching.
        
        Args:
            location (list): [latitude, longitude] for map center
            zoom_start (int): Initial zoom level
            
        Returns:
            folium.Map: Base map object
        """
        if location is None:
            location = _self.default_location
        if zoom_start is None:
            zoom_start = _self.default_zoom
            
        return folium.Map(location=location, zoom_start=zoom_start)
    
    def add_markers_to_map(self, map_obj, df_subset, highlight_top_n=0):
        """
        Add markers to the map with highlighting for top results.
        
        Args:
            map_obj (folium.Map): Map object to add markers to
            df_subset (pd.DataFrame): Data containing memories to display
            highlight_top_n (int): Number of top results to highlight in red
            
        Returns:
            folium.Map: Map object with added markers
        """
        for i, (_, row) in enumerate(df_subset.iterrows()):
            html = self.create_popup_html(row)
            iframe = folium.IFrame(html=html, width=290, height=350)
            popup = folium.Popup(iframe, max_width=290)

            # Highlight top N in red, others in blue
            color = "red" if i < highlight_top_n else "blue"

            folium.Marker(
                [float(row["latitude"]), float(row["longitude"])],
                popup=popup,
                tooltip=f"📍 {row['location']}",
                icon=folium.Icon(color=color, icon="heart")
            ).add_to(map_obj)
        
        return map_obj
    
    def create_search_results_map(self, results_df, highlight_count=3, bounds=None):
        """
        Create a map focused on search results.
        
        Args:
            results_df (pd.DataFrame): Search results data
            highlight_count (int): Number of top results to highlight
            bounds (list): Map bounds [[min_lat, min_lon], [max_lat, max_lon]]
            
        Returns:
            folium.Map: Map with search results
        """
        # Create base map
        map_obj = self.create_base_map()
        
        # Add markers for search results
        map_obj = self.add_markers_to_map(map_obj, results_df, highlight_count)
        
        # Fit bounds if provided
        if bounds and len(results_df) > 0:
            map_obj.fit_bounds(bounds)
        
        return map_obj
    
    def create_all_memories_map(self, all_memories_df):
        """
        Create a map showing all memories.
        
        Args:
            all_memories_df (pd.DataFrame): All memories data
            
        Returns:
            folium.Map: Map with all memories
        """
        # Create base map
        map_obj = self.create_base_map()
        
        # Add markers for all memories (no highlighting)
        map_obj = self.add_markers_to_map(map_obj, all_memories_df, highlight_top_n=0)
        
        return map_obj
    
    def display_image_grid(self, results_df, top_n=3):
        """
        Display a grid of images for the top search results.
        
        Args:
            results_df (pd.DataFrame): Search results data
            top_n (int): Number of images to display
        """
        if results_df.empty:
            return
        
        # Limit to top N results
        display_df = results_df.head(top_n)
        
        st.markdown("### 🖼️ Top Matching Memories")
        cols = st.columns(len(display_df))

        for i, (_, row) in enumerate(display_df.iterrows()):
            image_path = f"images/{row['image']}"
            with cols[i]:
                if os.path.exists(image_path):
                    st.image(
                        image_path, 
                        caption=f"{row['location']} — {row['caption']}", 
                        use_container_width=True
                    )
                else:
                    st.markdown(
                        f"📍 **{row['location']}**<br>{row['caption']}<br>*(Image not found)*", 
                        unsafe_allow_html=True
                    )
    
    def display_success_message(self, message):
        """
        Display a success message to the user.
        
        Args:
            message (str): Message to display
        """
        st.success(message)
    
    def display_info_message(self, message):
        """
        Display an info message to the user.
        
        Args:
            message (str): Message to display
        """
        st.info(message)
    
    def display_warning_message(self, message):
        """
        Display a warning message to the user.
        
        Args:
            message (str): Message to display
        """
        st.warning(message)
    
    def execute_action(self, decision: Dict[str, Any], results_df: pd.DataFrame, all_memories_df: pd.DataFrame):
        """
        Execute the action based on the decision.
        
        Args:
            decision (Dict[str, Any]): Decision dictionary from Decision module
            results_df (pd.DataFrame): Search results data
            all_memories_df (pd.DataFrame): All memories data
            
        Returns:
            folium.Map: The map object to display
        """
        action = decision.get('action', 'show_all_memories')
        message = decision.get('message', '')
        
        # Display appropriate message
        if decision.get('show_search_results', False):
            self.display_success_message(message)
        else:
            self.display_info_message(message)
        
        # Show image grid if needed
        if decision.get('show_search_results', False) and decision.get('action') == 'show_search_results':
            top_n = decision.get('top_n', 3)
            self.display_image_grid(results_df, top_n)
        
        # Create and return appropriate map
        if action == 'show_search_results':
            highlight_count = decision.get('highlight_count', 0)
            bounds = None
            if decision.get('zoom_to_results', False):
                from decision import Decision
                decision_module = Decision()
                bounds = decision_module.determine_map_bounds(results_df)
            
            return self.create_search_results_map(results_df, highlight_count, bounds)
        else:
            return self.create_all_memories_map(all_memories_df)
    
    def set_image_settings(self, max_width=400, max_height=300, quality=85):
        """
        Set image processing settings.
        
        Args:
            max_width (int): Maximum image width
            max_height (int): Maximum image height
            quality (int): JPEG quality (1-100)
        """
        self.image_max_width = max_width
        self.image_max_height = max_height
        self.image_quality = quality