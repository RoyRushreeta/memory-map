"""
📍 MEMORY MAP STREAMLIT APP - Main UI Interface

This is the main Streamlit app that provides the user interface for the Memory Map
with AI Search. It uses the Agent orchestrator to coordinate all backend modules
while maintaining the exact same visual behavior and functionality.
"""

import streamlit as st
from streamlit_folium import st_folium
from agent import Agent

# ======================
# INITIALIZE AGENT
# ======================

@st.cache_resource
def initialize_agent():
    """Initialize the Agent with caching to avoid reloading."""
    try:
        agent = Agent("memories.csv")
        return agent
    except Exception as e:
        st.error(f"Failed to initialize Memory Map Agent: {e}")
        return None


# ======================
# STREAMLIT UI
# ======================

st.set_page_config(page_title="Memory Map", layout="wide")
st.title("📍 Memory Map with AI Search")
st.markdown("Search and explore your memories using natural language 💛")

# Initialize the agent
agent = initialize_agent()
if agent is None:
    st.error("Failed to initialize the system. Please check your data files.")
    st.stop()

# Verify agent is ready
if not agent.is_ready():
    st.error("System not ready. Please check your memories.csv file and dependencies.")
    st.stop()

# Search bar
query = st.text_input("🔍 Ask something like 'Show my mountain trips' or 'Where did I click food photos?'")

# Process query or show all memories
if query and query.strip():
    # Use the agent to respond to the query
    map_obj, response_info = agent.respond_to_query(query.strip())
    
    # Display the map section header
    st.markdown("### 🗺️ Map of Top Memories")
else:
    # Show all memories when no query is provided
    map_obj = agent.get_all_memories_map()

# Display the map
st_data = st_folium(map_obj, width=725, height=500)