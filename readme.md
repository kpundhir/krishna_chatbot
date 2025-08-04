# Krishna Chatbot (Gita‑aligned QA Bot)

## Overview
**Krishna Chatbot** is a culturally grounded **NLP-based question-answering system** that leverages the **Bhagavad Gita** to respond to user queries with semantically relevant, scripture-aligned answers. It uses modern vector search and retrieval techniques to serve as a knowledge assistant for philosophical and spiritual questions.

---

## Features

- **Gita-based Responses**: Matches user queries to the most relevant verses from the Bhagavad Gita
- **Semantic Search**: Embeds 700+ verses into a vector database using `sentence-transformers` and retrieves contextually relevant matches
- **Retrieval-Augmented QA**: Combines FAISS vector search with contextual prompt templates for grounded response generation
- **LLM Generation**:  Uses a Hugging Face-hosted 7B model (e.g. zephyr-7b-alpha) to generate Krishna-style answers.
- **Modular Pipeline**:
  - Data chunking and verse preprocessing
  - Sentence-level embedding using pre-trained models
  - FAISS vector indexing and similarity search
- **Local Prototype**: Interact with a command-line interface (CLI) that returns surfaced passages and inferred answers

---

## Tech Stack

| Component     | Tool/Library            |
|---------------|-------------------------|
| Vector Search | FAISS via langchain_community                  |
| Embeddings    | HuggingFaceEmbeddings |
| NLP Framework | LangChain, transformers    |
| LLM Inference | transformers (zephyr-7b-alpha) |
| CLI Interface  | Python input() loop

## File Structure (Sample)
```bash
krishna_chatbot/
├── data/
│   └── gita_verses_cleaned.csv     # Source text: verse, chapter, translation
├── embeddings/
│   └── gita_faiss_index/           # Serialized FAISS index and metadata
├── app/
│   └── chatbot_langchain.py        # Main app file (CLI + RAG pipeline)
├── utils/                          # (Optional) Modular helpers for embedding/search
├── .env                            # Hugging Face token
└── README.md
```

## Getting Started
### 1. Clone the Repo
```bash
git clone https://github.com/yourusername/krishna-chatbot.git
cd krishna-chatbot
```

### 2. Set up Environment
```bash
HUGGINGFACE_TOKEN=your_hf_token_here #create a new .env file
pip install -r requirements.txt
pip install accelerate blobfile
```
Note: This script uses large LLMs hosted on Hugging Face Inference API. Expect latency and high resource consumption if run locally. Do not use on low-resource machines without modification.
### 3. Run ChatBot
```bash
python app/chatbot_langchain.py
```
## Example
**Input**:
Why is detachment important?

**Top Verse Retrieved**:
"You have a right to perform your prescribed duties, but you are not entitled to the fruits of your actions..." – Chapter 2, Verse 47

**Generated Response (Krishna-style)**:
One must act with commitment, Arjuna, yet remain free from attachment to the result. The wise find peace not in success or failure, but in the integrity of their action.

### Notes

This project showcases:
- Applied NLP: Retrieval-augmented generation with LangChain and Hugging Face
- MLOps Awareness: Model switching, API token handling, CLI integration
- Thematic NLP: Cultural + scriptural alignment with persona prompting

### Acknowledgements
Bhagavad Gita translations sourced from public domain interpretations
Models: HuggingFaceH4/zephyr-7b-alpha