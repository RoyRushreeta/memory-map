import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import base64
import os
from PIL import Image
import io
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# ======================
# IMAGE HANDLING
# ======================

@st.cache_data
def resize_and_encode_image(image_path, max_width=400, max_height=300, quality=85):
    """Resize image and convert to base64 string."""
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


def create_popup_html(row):
    """Create the HTML popup for each marker."""
    location = str(row.get("location", "Unknown Location")).strip()
    caption = str(row.get("caption", "No description available")).strip()
    image_name = str(row.get("image", "")).strip()

    image_path = f"images/{image_name}"
    base64_image = resize_and_encode_image(image_path)

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


# ======================
# DATA LOADING
# ======================

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("memories.csv")
        return df
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return pd.DataFrame()


@st.cache_resource
def create_faiss_index(df):
    """Create FAISS index from text embeddings."""
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Combine text fields
    texts = (df["location"].fillna('') + " " + df["caption"].fillna('')).tolist()
    embeddings = model.encode(texts, show_progress_bar=True)

    # Normalize for cosine similarity
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype('float32'))

    return model, index, embeddings


# ======================
# MAP CREATION
# ======================

@st.cache_data
def create_base_map():
    return folium.Map(location=[20.5937, 78.9629], zoom_start=5)


def add_markers_to_map(m, df_subset, highlight_top_n=0):
    """Add markers to the map. Highlight top N in red."""
    for i, (_, row) in enumerate(df_subset.iterrows()):
        html = create_popup_html(row)
        iframe = folium.IFrame(html=html, width=290, height=350)
        popup = folium.Popup(iframe, max_width=290)

        # Highlight top N in red, others in blue
        color = "red" if i < highlight_top_n else "blue"

        folium.Marker(
            [float(row["latitude"]), float(row["longitude"])],
            popup=popup,
            tooltip=f"📍 {row['location']}",
            icon=folium.Icon(color=color, icon="heart")
        ).add_to(m)
    return m


# ======================
# STREAMLIT UI
# ======================

st.set_page_config(page_title="Memory Map", layout="wide")
st.title("📍 Memory Map with AI Search")
st.markdown("Search and explore your memories using natural language 💛")

df = load_data()
if df.empty:
    st.error("No data found in memories.csv")
    st.stop()

model, index, embeddings = create_faiss_index(df)

# Search bar
query = st.text_input("🔍 Ask something like 'Show my mountain trips' or 'Where did I click food photos?'")

if query:
    query_emb = model.encode([query])
    query_emb = query_emb / np.linalg.norm(query_emb)
    scores, indices = index.search(query_emb.astype('float32'), k=10)

    matched_df = df.iloc[indices[0]]
    top_n = 3
    matched_df = matched_df.head(top_n)

    st.success(f"Showing top {top_n} similar memories for: '{query}'")

    # Show top 3 retrieved memories
    st.markdown("### 🖼️ Top Matching Memories")
    cols = st.columns(top_n)

    for i, (_, row) in enumerate(matched_df.iterrows()):
        image_path = f"images/{row['image']}"
        with cols[i]:
            if os.path.exists(image_path):
                st.image(image_path, caption=f"{row['location']} — {row['caption']}", use_container_width=True)
            else:
                st.markdown(f"📍 **{row['location']}**<br>{row['caption']}<br>*(Image not found)*", unsafe_allow_html=True)

    # 🗺️ Highlight top 3 on map and zoom in
    st.markdown("### 🗺️ Map of Top Memories")
    m = create_base_map()
    m = add_markers_to_map(m, matched_df, highlight_top_n=3)

    # Compute zoom bounds around the top 3 results
    bounds = [
        [matched_df["latitude"].min(), matched_df["longitude"].min()],
        [matched_df["latitude"].max(), matched_df["longitude"].max()],
    ]
    m.fit_bounds(bounds)

else:
    m = create_base_map()
    m = add_markers_to_map(m, df)

# Display map
st_data = st_folium(m, width=725, height=500)