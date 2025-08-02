# scripts/build_index.py

import os
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Define absolute base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "bhagavad_gita_cleaned.csv")
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "embeddings")
INDEX_SAVE_PATH = os.path.join(EMBEDDINGS_DIR, "gita_faiss.index")
TEXT_SAVE_PATH = os.path.join(EMBEDDINGS_DIR, "citation_texts.npy")

def main():
    print("🔹 Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    texts = df["citation_text"].tolist()

    print("🔹 Embedding texts...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, show_progress_bar=True)

    print("🔹 Building FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

    print("🔹 Saving index and texts...")
    faiss.write_index(index, INDEX_SAVE_PATH)
    np.save(TEXT_SAVE_PATH, texts)

    print("✅ Index and texts saved successfully.")
    print(f"    FAISS index → {INDEX_SAVE_PATH}")
    print(f"    Texts file  → {TEXT_SAVE_PATH}")

if __name__ == "__main__":
    main()
