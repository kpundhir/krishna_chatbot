import os
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from huggingface_hub import InferenceClient

load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

faiss_index = FAISS.load_local(
    "embeddings/gita_faiss_index",
    embeddings=embedding_model,
    allow_dangerous_deserialization=True
)

model_name = "mistralai/Mistral-7B-Instruct-v0.1"
client = InferenceClient(model=model_name, token=HF_TOKEN)

def get_top_verse(query, k=1):
    docs = faiss_index.similarity_search(query, k=k)
    return docs[0].page_content if docs else "No verse found."

def ask_krishna(query, retrieved_verse):
    prompt = f"""
You are Lord Krishna explaining the Bhagavad Gita to Arjuna.
Speak in a calm, wise, and spiritually grounded tone.
Use the following verse as your base for the answer.

Verse: {retrieved_verse}

Question: {query}

Answer:
    """
    try:
        response = client.text_generation(
            prompt,
            max_new_tokens=200,
            temperature=0.7,
            repetition_penalty=1.1
        )
        if not response.strip():
            return "Model returned an empty response."
        return response.strip().split("Answer:")[-1].strip()
    except Exception as e:
        return f"HuggingFace Inference API failed: {e}"

if __name__ == "__main__":
    print("Welcome to the Krishna Chatbot.")
    while True:
        try:
            user_input = input("\nYour Question (or 'exit'): ")
            if user_input.lower() in ["exit", "quit"]:
                print("Jai Shri Krishna!")
                break

            top_verse = get_top_verse(user_input)
            print(f"\nClosest Verse: {top_verse}")

            response = ask_krishna(user_input, top_verse)
            print(f"\nKrishna's Response:\n{response}")

        except Exception as e:
            print(f"Error: {e}")
