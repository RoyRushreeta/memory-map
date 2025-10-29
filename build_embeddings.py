# build_embeddings.py
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

# Load data
df = pd.read_csv("memories.csv")

# Combine meaningful text (location + caption)
texts = (df['location'].fillna('') + ' ' + df['caption'].fillna('')).tolist()

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Create embeddings
embeddings = model.encode(texts, convert_to_numpy=True)

# Create FAISS index
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

# Save FAISS index and dataframe mapping
faiss.write_index(index, "memory_index.faiss")
with open("memory_texts.pkl", "wb") as f:
    pickle.dump(df, f)

print(f"✅ Saved FAISS index with {len(df)} memories.")