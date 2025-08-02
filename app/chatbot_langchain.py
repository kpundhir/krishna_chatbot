import os
import numpy as np
import faiss
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore

# --- Load environment variables ---
load_dotenv()

# --- Load citation texts and FAISS index ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "embeddings")
TEXTS_PATH = os.path.join(EMBEDDINGS_DIR, "citation_texts.npy")
INDEX_PATH = os.path.join(EMBEDDINGS_DIR, "gita_faiss.index")

citation_texts = np.load(TEXTS_PATH, allow_pickle=True).tolist()
faiss_index = faiss.read_index(INDEX_PATH)

# --- Embedding model ---
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# --- Reconstruct document store ---
documents = [Document(page_content=text) for text in citation_texts]
docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(documents)})
index_to_docstore_id = {i: str(i) for i in range(len(documents))}

# --- Vectorstore ---
vectorstore = FAISS(
    index=faiss_index,
    docstore=docstore,
    index_to_docstore_id=index_to_docstore_id,
    embedding_function=embedding_model
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# --- Inference Loop ---
def main():
    print("📿 Krishna Retrieval Tool (No LLM). Ask your question (type 'exit' to quit).")
    while True:
        query = input("\nYour question: ")
        if query.lower() in ["exit", "quit"]:
            print("🙏 Thank you. May Krishna's wisdom guide you.")
            break
        docs = retriever.get_relevant_documents(query)
        print("\n🔍 Top Matching Verses:\n")
        for i, doc in enumerate(docs, 1):
            print(f"{i}. {doc.page_content}\n")

if __name__ == "__main__":
    main()
