# Aceso — Medical AI Chatbot

> An intelligent medical information assistant powered by ReAct (Reasoning + Acting) architecture with real-time web search and vector database retrieval.

**Live Demo:** [https://huggingface.co/spaces/jacob03/medical_chatbot](https://huggingface.co/spaces/jacob03/medical_chatbot)

---

## Overview

Aceso is a medical chatbot that answers health and medical questions by reasoning through problems step-by-step, searching a curated medical knowledge base, and fetching the latest guidelines from the web. It shows its full reasoning process so users can understand how answers are derived.

### Key Features

- **ReAct Architecture** — Explicit Thought → Action → Observation reasoning loop powered by LangGraph
- **Dual Search** — Searches a Pinecone vector database (Gale Encyclopedia of Medicine) for established facts, and DuckDuckGo for current guidelines, outbreaks, and recent data
- **Medical Domain Filtering** — Web search prioritises trusted sources (WHO, NIH, CDC, PubMed, Mayo Clinic, WebMD)
- **CrossEncoder Reranking** — Retrieved chunks are reranked with `BAAI/bge-reranker-large` for relevance
- **Safety Classifier** — Every response is checked before being shown to the user
- **Conversation History** — Maintains context across turns with automatic summarisation for long sessions
- **Collapsible Reasoning Trace** — Users can show or hide the full reasoning process

---

## Architecture

```
User Question
      │
      ▼
 ReAct Agent (LangGraph StateGraph)
      │
      ├── Thought: Analyse what information is needed
      │
      ├── Action: search_medical_database  ──► Pinecone Vector Store
      │          (established facts)             + CrossEncoder Reranker
      │
      ├── Action: search_web_medical  ──────► DuckDuckGo (medical domains)
      │          (current data)
      │
      ├── Thought: Synthesise findings
      │
      └── Action: Finish ──► Safety Check ──► Final Answer
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Web Framework | Flask + Gunicorn |
| LLM | Groq API (`llama-3.3-70b-versatile`) |
| Orchestration | LangGraph (ReAct StateGraph) |
| Vector Database | Pinecone |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Reranker | `BAAI/bge-reranker-large` (CrossEncoder) |
| Web Search | DuckDuckGo Search (DDGS) |
| Safety Model | Groq API (`llama-3.1-8b-instant`) |
| History Summarisation | Groq API (`llama-3.1-8b-instant`) |
| Deployment | Hugging Face Spaces (Docker) |

---

## Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/jacobjk03/Medical_chatbot.git
cd Medical_chatbot
```

### 2. Create and activate a conda environment

```bash
conda create -n medicalchatbot python=3.10 -y
conda activate medicalchatbot
```

### 3. Install dependencies

```bash
# Install CPU-only torch first (avoids downloading the 2.5GB GPU build)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining requirements
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file in the root directory:

```ini
GROQ_API_KEY=your_groq_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
FLASK_SECRET_KEY=any_random_secret_string
```

Get your keys:
- Groq API key: [console.groq.com](https://console.groq.com)
- Pinecone API key: [app.pinecone.io](https://app.pinecone.io)

### 5. Ingest the medical knowledge base (first time only)

```bash
python store_index.py
```

This embeds the Gale Encyclopedia of Medicine PDF into Pinecone. Skip if the index already exists.

### 6. Run the app

```bash
python app.py
```

Open [http://localhost:7860](http://localhost:7860)

---

## Docker

```bash
# Build
docker build -t aceso-medical-ai .

# Run
docker run -p 7860:7860 \
  -e GROQ_API_KEY=your_key \
  -e PINECONE_API_KEY=your_key \
  -e FLASK_SECRET_KEY=your_secret \
  aceso-medical-ai
```

---

## Usage

| Command | Description |
|---|---|
| Ask any medical question | e.g. "What are the symptoms of diabetes?" |
| `reasoning on` / `reasoning off` | Toggle the reasoning trace display |
| `reset` | Clear conversation history |
| `help` | Show available commands |

**Disclaimer:** This chatbot is for educational purposes only. Always consult a qualified healthcare professional for personal medical advice.

---

## Deployment

Deployed on **Hugging Face Spaces** with Docker. The Space automatically rebuilds on every push to the `main` branch via GitHub sync.

Environment variables are configured as Space secrets (never committed to the repo).
