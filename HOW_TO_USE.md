# 🧠 DontForget: Neural Database Engine

**DontForget** is a local neural database for your terminal, implementing the hybrid architecture from *"Neural Databases Using Large Language Models"*. It combines SPO triplet storage, vector embeddings, keyword search, and a Text-to-SQL generator — all powered by OpenAI.

## ✨ Features

* **SPO Triplet Storage** — Every memory is decomposed into Subject-Predicate-Object triplets for structured knowledge retrieval.
* **Hybrid Search** — Combines OpenAI `text-embedding-3-large` vector similarity with SQLite FTS5 keyword search, merged by combined score.
* **Knowledge Editing with Gating** — Edit stored facts without reindexing. Uses the gating mechanism `gate = σ(sim(q,k)/τ)` to override stale results.
* **T2SQL Generator** — Ask natural language questions against any SQLite database. Uses few-shot Chain-of-Thought prompting.
* **LLM Reranking** — Top results are reranked by the LLM for precision.
* **Zero-Friction CLI** — `mem r`, `mem q`, `mem d`, `mem e`, `mem sql`.

## 🚀 Setup

### 1. Prerequisites

* Python 3.10+
* An OpenAI API Key

### 2. Install

```bash
cd dontforget
python3 -m venv venv
source venv/bin/activate
pip install fastapi uvicorn python-dotenv openai pydantic numpy
```

### 3. Configure

```bash
cp .env.example .env
# Edit .env with your keys:
#   OPENAI_API_KEY="sk-..."
#   DONTFORGET_SECRET_KEY="your_secret"
```

### 4. Run Server

```bash
source venv/bin/activate
python main.py
# Server runs on http://0.0.0.0:8000
```

### 5. Setup CLI

```bash
cp mem-cli /usr/local/bin/mem
chmod +x /usr/local/bin/mem
# Add to ~/.bashrc or ~/.zshrc:
export DONTFORGET_SECRET_KEY="your_secret"
export DONTFORGET_API_URL="http://0.0.0.0:8000"
```

## 📖 Usage

### Remember
```bash
mem r "Paid 432 rs to Akash for dinner"
# ✔ Saved! Tags: finance, debt, akash
# SPO: (I) —[paid]→ (432 rs to Akash for dinner)
```

### Query
```bash
mem q "How much do I owe Akash?"
# Uses hybrid vector+keyword search → reranking → LLM synthesis
```

### Delete
```bash
mem d "That note about Akash"
```

### Edit a Fact (Gating Mechanism)
```bash
mem e "How much I owe Akash" "I paid Akash back, debt is cleared"
# Future queries about Akash debt will return the edited fact
```

### Text-to-SQL
```bash
mem sql "How many users signed up last month?" "/path/to/app.db"
# 💭 Thought: Count users with created_at in last month...
# 📝 SQL: SELECT COUNT(*) FROM users WHERE created_at >= date('now', '-1 month')
```

## 🏗️ Architecture

Based on the paper's hybrid neural database framework:

1. **Ingestion**: Text → LLM extracts tags + SPO triplets → OpenAI embedding generated → All stored in SQLite.
2. **Retrieval**: Query → keyword extraction → hybrid search (FTS5 + cosine similarity) → gating edits applied → LLM reranking → LLM synthesis.
3. **T2SQL**: Question → schema extraction → few-shot CoT prompting → SQL generation → execution.
4. **Knowledge Editing**: `gate = σ(sim(q,k)/τ)` — edited facts override matching retrievals without full reindex.

## 🛡️ API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/remember` | POST | Store a memory with tags, SPO triplets, and embedding |
| `/remind` | POST | Query memories using hybrid search + synthesis |
| `/delete` | POST | Delete memories by natural language description |
| `/edit` | POST | Store a knowledge edit (gating mechanism) |
| `/t2sql` | POST | Natural language → SQL against any SQLite DB |

---

**License:** GPL-3
**Author:** Suraj Kushwah
