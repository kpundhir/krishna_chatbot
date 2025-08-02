# scripts/search_interface.py

import numpy as np
import faiss
import os
from sentence_transformers import SentenceTransformer

# Paths
INDEX_PATH = os.path.join("..", "embeddings", "gita_faiss.index")
TEXT_PATH = os.path.join("..", "embeddings", "citation_texts.npy")

def load_index():
    index = faiss.read_index(INDEX_PATH)
    texts = np.load(TEXT_PATH, allow_pickle=True)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return index, texts, model

def search(query, top_k=3):
    index, texts, model = load_index()
    query_vec = model.encode([query])
    D, I = index.search(np.array(query_vec), top_k)
    print("\n🔍 Top Matches:")
    for i in I[0]:
        print("\n---")
        print(texts[i])

if __name__ == "__main__":
    while True:
        query = input("\nAsk a question (or type 'exit'): ")
        if query.lower() == 'exit':
            break
        search(query)
