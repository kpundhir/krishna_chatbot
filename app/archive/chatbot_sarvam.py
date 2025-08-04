import os
import numpy as np
import faiss
import requests
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings  # ✅ updated import
from langchain_community.docstore.in_memory import InMemoryDocstore

# Load environment variables
load_dotenv()
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")

# Initialize embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

faiss_index = FAISS.load_local(
    "embeddings/gita_faiss_index",  # ✅ CORRECT — matches where build_index.py saved it
    embeddings=embedding_model,
    allow_dangerous_deserialization=True
)

# Function to query FAISS
def get_top_verse(query, k=1):
    docs = faiss_index.similarity_search(query, k=k)
    return docs[0].page_content if docs else "No verse found."

def ask_sarvam_krishna(query, retrieved_verse):
    prompt = f"""
You are Lord Krishna explaining the Bhagavad Gita to Arjuna.
Speak in a calm, wise, and spiritually grounded tone.
Use the following verse as your base for the answer.

Verse: {retrieved_verse}

Question: {query}

Answer:
    """

    headers = {
        "Authorization": f"Bearer {SARVAM_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "prompt": prompt,
        "max_tokens": 200,
        "temperature": 0.7,
        "top_p": 0.9,
    }

    # 🔍 Debug print
    print("🟡 Sending request to Sarvam API...")
    print(f"🔹 Endpoint: https://api.sarvam.ai/v1/completions")
    print(f"🔹 API Key present: {'Yes' if SARVAM_API_KEY else 'No'}")
    print(f"🔹 Prompt preview:\n{prompt[:500]}...")  # first 500 chars
    print(f"🔹 Request payload: {data}")

    response = requests.post("https://api.sarvam.ai/v1/completions", headers=headers, json=data)

    if response.status_code != 200:
        print(f"🔴 API Error {response.status_code}: {response.text}")  # full error body
        raise Exception(f"Sarvam API error {response.status_code}: {response.text}")

    return response.json()['choices'][0]['text'].strip()


# CLI loop (optional, for quick test)
if __name__ == "__main__":
    print("🕉️ Welcome to the Krishna Chatbot. Ask your question below.")
    while True:
        try:
            user_input = input("\n🗣️ Your Question (or 'exit'): ")
            if user_input.lower() in ["exit", "quit"]:
                print("🙏 Jai Shri Krishna!")
                break

            top_verse = get_top_verse(user_input)
            print(f"\n📜 Closest Verse: {top_verse}")

            response = ask_sarvam_krishna(user_input, top_verse)
            print(f"\n🧘 Krishna's Response:\n{response}")

        except Exception as e:
            print(f"⚠️ Error: {e}")
