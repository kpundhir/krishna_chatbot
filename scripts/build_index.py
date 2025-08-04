import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "bhagavad_gita_cleaned.csv")
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "embeddings")
TEXT_SAVE_PATH = os.path.join(EMBEDDINGS_DIR, "citation_texts.npy")
LC_FAISS_SAVE_PATH = os.path.join(EMBEDDINGS_DIR, "gita_faiss_index")

def main():
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    texts = df["citation_text"].tolist()
    np.save(TEXT_SAVE_PATH, texts)

    print("Creating documents...")
    documents = [Document(page_content=text) for text in texts]

    print("Loading embedding model...")
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print("Building FAISS vectorstore...")
    faiss_db = FAISS.from_documents(documents, embedding=embedding_model)

    print("Saving vectorstore...")
    os.makedirs(LC_FAISS_SAVE_PATH, exist_ok=True)
    faiss_db.save_local(LC_FAISS_SAVE_PATH)

    print(f"Saved FAISS vectorstore to {LC_FAISS_SAVE_PATH}")
    print(f"Saved citation texts to {TEXT_SAVE_PATH}")

if __name__ == "__main__":
    main()
