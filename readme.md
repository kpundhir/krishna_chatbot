# Krishna Chatbot (Gita‑aligned QA Bot)

## Overview
**Krishna Chatbot** is a culturally grounded **NLP-based question-answering system** that leverages the **Bhagavad Gita** to respond to user queries with semantically relevant, scripture-aligned answers. It uses modern vector search and retrieval techniques to serve as a knowledge assistant for philosophical and spiritual questions.

---

## Features

- **Gita-based Responses**: Matches user queries to the most relevant verses from the Bhagavad Gita
- **Semantic Search**: Embeds 700+ verses into a vector database using `sentence-transformers` and retrieves contextually relevant matches
- **Retrieval-Augmented QA**: Combines FAISS vector search with contextual prompt templates for grounded response generation
- **Modular Pipeline**:
  - Data chunking and verse preprocessing
  - Sentence-level embedding using pre-trained models
  - FAISS vector indexing and similarity search
- **Local Prototype**: Interact with a command-line interface (CLI) that returns surfaced passages and inferred answers

---

## Tech Stack

| Component       | Tool/Library            |
|----------------|-------------------------|
| Vector Search   | FAISS                   |
| Embeddings      | `sentence-transformers` (e.g., `all-MiniLM-L6-v2`) |
| NLP Framework   | LangChain (optional)    |
| Language Models | Sarvam LLMs / HuggingFace Transformers |
| CLI Interface

## File Structure (Sample)
```bash
krishna_chatbot/
├── data/
│ └── gita_verses_cleaned.csv # Source text: verse, chapter, translation
├── utils/
│ ├── embed.py # Embedding logic + FAISS indexing
│ └── search.py # Query embedding + top-k search
├── cli.py # Command-line chatbot interface
├── app.py # Entry point
└── README.md
```

## Getting Started
### 1. Clone the Repo
```bash
git clone https://github.com/yourusername/krishna-chatbot.git
cd krishna-chatbot
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run CLI Bot
```bash
python app.py
```

