"""
DontForget — Neural Database Implementation
Based on: "Neural Databases Using Large Language Models" (Kamal, Sahil, Sahil Singh Ranout)

Architecture:
  1. Neural Engine: SPO triplet storage + hybrid vector/keyword retrieval + gating edits
  2. T2SQL Generator: Few-shot CoT natural language → SQL translation
  3. Query Router: Neural Engine first, T2SQL fallback
"""

import sqlite3
import datetime
import os
import json
import re
import struct
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel
from dotenv import load_dotenv
import numpy as np
from openai import OpenAI

# --- CONFIGURATION ---
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
SERVER_SECRET = os.getenv("DONTFORGET_SECRET_KEY")
DB_PATH = "memory.db"
EMBEDDING_MODEL = "text-embedding-3-large"  # 3072 dims per paper
EMBEDDING_DIMS = 3072
REWRITE_MODEL = "gpt-4o-mini"   # rewrite/extraction per paper
SYNTHESIS_MODEL = "gpt-4o-mini"  # synthesis (using 4o-mini to keep costs low; swap to gpt-4 if needed)
TEMPERATURE_TAU = 0.07  # gating temperature τ from Eq. 1

if not OPENAI_KEY or not SERVER_SECRET:
    raise ValueError("Set OPENAI_API_KEY and DONTFORGET_SECRET_KEY in .env")

app = FastAPI(title="DontForget — Neural Database")
oai = OpenAI(api_key=OPENAI_KEY)

# --- SECURITY ---
async def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != SERVER_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")

# --- MODELS ---
class ThoughtRequest(BaseModel):
    text: str

class QueryRequest(BaseModel):
    question: str

class EditRequest(BaseModel):
    query: str       # what to find
    new_value: str   # replacement content

class T2SQLRequest(BaseModel):
    question: str
    db_path: str = ""  # path to external SQLite DB

# --- EMBEDDING HELPERS ---

def _pack_embedding(vec: List[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)

def _unpack_embedding(blob: bytes) -> np.ndarray:
    n = len(blob) // 4
    return np.array(struct.unpack(f"{n}f", blob), dtype=np.float32)

def get_embedding(text: str) -> List[float]:
    resp = oai.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return resp.data[0].embedding

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))

# --- DATABASE INIT ---

def init_db():
    conn = sqlite3.connect(DB_PATH)

    # Main memories table (raw text + AI tags, same as before)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            raw_text TEXT,
            ai_tags TEXT
        )
    """)

    # FTS5 index for keyword search
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS memories_idx
        USING fts5(raw_text, ai_tags, content='memories', content_rowid='id')
    """)
    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
          INSERT INTO memories_idx(rowid, raw_text, ai_tags) VALUES (new.id, new.raw_text, new.ai_tags);
        END
    """)
    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
          INSERT INTO memories_idx(memories_idx, rowid, raw_text, ai_tags) VALUES('delete', old.id, old.raw_text, old.ai_tags);
        END
    """)

    # SPO triplets table (Neural Engine core)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS spo_triplets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            memory_id INTEGER REFERENCES memories(id) ON DELETE CASCADE,
            subject TEXT,
            predicate TEXT,
            object TEXT
        )
    """)

    # Embeddings table (HNSW approximated via brute-force cosine on stored vectors)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            memory_id INTEGER PRIMARY KEY REFERENCES memories(id) ON DELETE CASCADE,
            vector BLOB
        )
    """)

    # Knowledge edits KV store (gating mechanism from paper Eq. 1)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS knowledge_edits (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query_embedding BLOB,
            new_value TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()

init_db()

# --- LLM HELPERS ---

def llm_json(prompt: str, model: str = REWRITE_MODEL) -> dict:
    resp = oai.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)

def llm_text(prompt: str, model: str = SYNTHESIS_MODEL) -> str:
    resp = oai.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content

# --- SEARCH ENGINE ---

def keyword_search(keywords: List[str], limit: int = 30) -> List[dict]:
    """FTS5 keyword search — AND first, fallback to OR."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    clean = [re.sub(r'[^a-zA-Z0-9]', '', k) for k in keywords if k]
    if not clean:
        conn.close()
        return []

    rows = []
    # AND
    try:
        q = " AND ".join(f'"{k}"' for k in clean)
        cur = conn.execute(
            "SELECT rowid FROM memories_idx WHERE memories_idx MATCH ? ORDER BY rank LIMIT ?",
            (q, limit),
        )
        rows = [r["rowid"] for r in cur.fetchall()]
    except Exception:
        pass

    # OR fallback
    if len(rows) < 5:
        try:
            q = " OR ".join(f'"{k}"' for k in clean)
            cur = conn.execute(
                "SELECT rowid FROM memories_idx WHERE memories_idx MATCH ? ORDER BY rank LIMIT ?",
                (q, limit),
            )
            ids_set = set(rows)
            for r in cur.fetchall():
                if r["rowid"] not in ids_set:
                    rows.append(r["rowid"])
        except Exception:
            pass

    if not rows:
        conn.close()
        return []

    placeholders = ",".join("?" * len(rows))
    cur = conn.execute(
        f"SELECT id, raw_text, ai_tags, timestamp FROM memories WHERE id IN ({placeholders})",
        rows,
    )
    results = [dict(r) for r in cur.fetchall()]
    conn.close()
    return results


def vector_search(query_embedding: np.ndarray, top_k: int = 30) -> List[dict]:
    """Brute-force cosine similarity over stored embeddings (approximates HNSW)."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.execute("SELECT memory_id, vector FROM embeddings")
    scored = []
    for row in cur.fetchall():
        vec = _unpack_embedding(row["vector"])
        sim = cosine_sim(query_embedding, vec)
        scored.append((row["memory_id"], sim))
    scored.sort(key=lambda x: x[1], reverse=True)
    top_ids = [mid for mid, _ in scored[:top_k]]

    if not top_ids:
        conn.close()
        return []

    placeholders = ",".join("?" * len(top_ids))
    cur = conn.execute(
        f"SELECT id, raw_text, ai_tags, timestamp FROM memories WHERE id IN ({placeholders})",
        top_ids,
    )
    results = [dict(r) for r in cur.fetchall()]
    # attach scores
    score_map = dict(scored[:top_k])
    for r in results:
        r["_vscore"] = score_map.get(r["id"], 0)
    conn.close()
    return results


def hybrid_search(keywords: List[str], query_embedding: np.ndarray, limit: int = 30) -> List[dict]:
    """Hybrid retrieval: merge keyword + vector results, deduplicate, rank by combined score."""
    kw_results = keyword_search(keywords, limit)
    vec_results = vector_search(query_embedding, limit)

    merged = {}
    # keyword results get a base score
    for i, r in enumerate(kw_results):
        mid = r["id"]
        merged[mid] = r
        merged[mid]["_score"] = (limit - i) / limit  # rank-based score

    # vector results add their cosine similarity
    for r in vec_results:
        mid = r["id"]
        if mid in merged:
            merged[mid]["_score"] += r.get("_vscore", 0)
        else:
            merged[mid] = r
            merged[mid]["_score"] = r.get("_vscore", 0)

    ranked = sorted(merged.values(), key=lambda x: x.get("_score", 0), reverse=True)
    return ranked[:limit]


def apply_knowledge_edits(query_embedding: np.ndarray, results: List[dict]) -> List[dict]:
    """Gating mechanism from paper Eq. 1: gate = σ(sim(q,k)/τ).
    If gate > 0.5, the edited fact overrides the retrieval result."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    edits = conn.execute("SELECT id, query_embedding, new_value FROM knowledge_edits").fetchall()
    conn.close()

    if not edits:
        return results

    for edit in edits:
        edit_emb = _unpack_embedding(edit["query_embedding"])
        sim = cosine_sim(query_embedding, edit_emb)
        gate = sigmoid(sim / TEMPERATURE_TAU)
        if gate > 0.5:
            # Override the most relevant result with the edited value
            if results:
                results[0]["raw_text"] = edit["new_value"]
                results[0]["_edited"] = True
            else:
                results.append({
                    "id": -1,
                    "raw_text": edit["new_value"],
                    "ai_tags": "edited",
                    "timestamp": str(datetime.datetime.now()),
                    "_edited": True,
                })
    return results


# --- SPO TRIPLET EXTRACTION ---

def extract_spo(text: str) -> List[dict]:
    """Extract Subject-Predicate-Object triplets from text using LLM."""
    data = llm_json(f"""Extract Subject-Predicate-Object triplets from this text.
Text: "{text}"
Return JSON: {{"triplets": [{{"subject":"...","predicate":"...","object":"..."}}]}}
Return 1-3 triplets. Keep them concise.""")
    return data.get("triplets", [])


# --- T2SQL GENERATOR (Few-shot CoT) ---

T2SQL_SYSTEM = """You are a Text-to-SQL translator. Given a natural language question and a database schema, produce a valid SQLite SQL query.

Use Chain-of-Thought reasoning:
1. Identify which tables and columns are relevant.
2. Determine the SQL operation (SELECT, JOIN, WHERE, GROUP BY).
3. Write the SQL query.

Few-shot examples:

Q: "How many users signed up last month?"
Schema: users(id, name, email, created_at)
Thought: Need to count users where created_at is within last month. Use COUNT and date filter.
SQL: SELECT COUNT(*) FROM users WHERE created_at >= date('now', '-1 month')

Q: "What are the top 5 products by revenue?"
Schema: orders(id, product_id, amount, created_at), products(id, name, price)
Thought: Need to join orders with products, sum the amount per product, order descending, limit 5.
SQL: SELECT p.name, SUM(o.amount) as revenue FROM orders o JOIN products p ON o.product_id = p.id GROUP BY p.name ORDER BY revenue DESC LIMIT 5

Q: "Show me all employees in the engineering department"
Schema: employees(id, name, department_id), departments(id, name)
Thought: Join employees with departments, filter where department name is engineering.
SQL: SELECT e.name FROM employees e JOIN departments d ON e.department_id = d.id WHERE d.name = 'engineering'

Return ONLY JSON: {"thought": "...", "sql": "..."}"""


def generate_sql(question: str, schema: str) -> dict:
    resp = oai.chat.completions.create(
        model=REWRITE_MODEL,
        messages=[
            {"role": "system", "content": T2SQL_SYSTEM},
            {"role": "user", "content": f"Question: {question}\nSchema: {schema}"},
        ],
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)


def get_schema(db_path: str) -> str:
    """Extract schema from an external SQLite database."""
    conn = sqlite3.connect(db_path)
    cur = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL")
    schema = "\n".join(row[0] for row in cur.fetchall())
    conn.close()
    return schema


# --- RERANKING (LLM-based, replaces cross-encoder) ---

def rerank(query: str, results: List[dict], top_k: int = 10) -> List[dict]:
    if len(results) <= top_k:
        return results
    # Build candidates
    candidates = "\n".join(
        f"[{i}] {r['raw_text'][:200]}" for i, r in enumerate(results)
    )
    data = llm_json(f"""Rerank these search results by relevance to the query.
Query: "{query}"
Candidates:
{candidates}
Return JSON: {{"ranked_indices": [0, 3, 1, ...]}} — list of indices from most to least relevant. Return top {top_k}.""")
    indices = data.get("ranked_indices", list(range(top_k)))
    return [results[i] for i in indices if i < len(results)][:top_k]


# --- ENDPOINTS ---

@app.post("/remember", dependencies=[Depends(verify_api_key)])
def remember(request: ThoughtRequest):
    try:
        # 1. AI Tagging
        tags_data = llm_json(f"""Generate 5 search tags for this thought.
Input: "{request.text}"
Return JSON: {{"tags": ["tag1", "tag2"]}}""")
        tags = tags_data.get("tags", [])
        tags_str = ", ".join(tags)

        # 2. SPO Triplet extraction
        triplets = extract_spo(request.text)

        # 3. Generate embedding
        emb = get_embedding(request.text)

        # 4. Store everything
        conn = sqlite3.connect(DB_PATH)
        cur = conn.execute(
            "INSERT INTO memories (raw_text, ai_tags) VALUES (?, ?)",
            (request.text, tags_str),
        )
        mem_id = cur.lastrowid

        # Store embedding
        conn.execute(
            "INSERT INTO embeddings (memory_id, vector) VALUES (?, ?)",
            (mem_id, _pack_embedding(emb)),
        )

        # Store SPO triplets
        for t in triplets:
            conn.execute(
                "INSERT INTO spo_triplets (memory_id, subject, predicate, object) VALUES (?, ?, ?, ?)",
                (mem_id, t.get("subject", ""), t.get("predicate", ""), t.get("object", "")),
            )

        conn.commit()
        conn.close()

        print(f"🧠 Saved: {request.text[:40]}... [Tags: {tags_str}] [SPO: {len(triplets)}]")
        return {"status": "saved", "tags": tags, "triplets": triplets}

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(500, str(e))


@app.post("/remind", dependencies=[Depends(verify_api_key)])
def remind(request: QueryRequest):
    try:
        today = datetime.datetime.now().strftime("%Y-%m-%d %A")

        # 1. Extract keywords (rewrite model)
        kw_data = llm_json(f"""Extract 3-5 search keywords from this query.
Query: "{request.question}"
Return JSON: {{"keywords": ["word1", "word2"]}}""")
        keywords = kw_data.get("keywords", [])
        print(f"🕵️ Keywords: {keywords}")

        # 2. Get query embedding
        query_emb = np.array(get_embedding(request.question), dtype=np.float32)

        # 3. Hybrid search (vector + keyword)
        results = hybrid_search(keywords, query_emb, limit=30)

        # 4. Apply knowledge edits (gating mechanism)
        results = apply_knowledge_edits(query_emb, results)

        # 5. Rerank
        if len(results) > 10:
            results = rerank(request.question, results, top_k=10)

        # 6. Build context
        context_str = ""
        for r in results:
            edited = " [EDITED]" if r.get("_edited") else ""
            context_str += f"[ID:{r['id']}] [{r.get('timestamp','')}]{edited} {r['raw_text']}\n"

        # Also fetch relevant SPO triplets
        if results:
            conn = sqlite3.connect(DB_PATH)
            ids = [r["id"] for r in results if r["id"] > 0]
            if ids:
                ph = ",".join("?" * len(ids))
                cur = conn.execute(
                    f"SELECT memory_id, subject, predicate, object FROM spo_triplets WHERE memory_id IN ({ph})",
                    ids,
                )
                spo_context = "\n".join(
                    f"  ({row[1]}) —[{row[2]}]→ ({row[3]})" for row in cur.fetchall()
                )
                if spo_context:
                    context_str += f"\nKnowledge Graph Triplets:\n{spo_context}\n"
            conn.close()

        token_est = len(context_str) // 4

        # 7. Synthesize answer (synthesis model)
        answer = llm_text(f"""You are a Memory Assistant. Date: {today}
User Query: "{request.question}"

Relevant Memories:
{context_str}

Instructions:
1. Answer using ONLY the memories above.
2. Check timestamps for time-relative queries ("today", "last week").
3. Use the Knowledge Graph Triplets for relationship queries.
4. If no relevant memories, say "No relevant info found."
""")

        return {
            "answer": answer,
            "stats": {"found": len(results), "tokens": token_est},
        }

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(500, str(e))


@app.post("/delete", dependencies=[Depends(verify_api_key)])
def delete_endpoint(request: QueryRequest):
    try:
        keywords = request.question.split()
        query_emb = np.array(get_embedding(request.question), dtype=np.float32)
        results = hybrid_search(keywords, query_emb, limit=10)

        if not results:
            return {"answer": "No items found."}

        context = "\n".join(f"ID:{r['id']} Text:{r['raw_text']}" for r in results)
        data = llm_json(f"""User wants to delete: "{request.question}"
Which IDs match?
Options:
{context}
Return JSON: {{"ids": [1]}} or {{"ids": []}}""")
        ids = data.get("ids", [])

        if ids:
            conn = sqlite3.connect(DB_PATH)
            ph = ",".join("?" * len(ids))
            conn.execute(f"DELETE FROM spo_triplets WHERE memory_id IN ({ph})", ids)
            conn.execute(f"DELETE FROM embeddings WHERE memory_id IN ({ph})", ids)
            conn.execute(f"DELETE FROM memories WHERE id IN ({ph})", ids)
            conn.commit()
            conn.close()
            return {"answer": f"Deleted {len(ids)} items."}
        return {"answer": "Found items, but none matched exactly."}

    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/edit", dependencies=[Depends(verify_api_key)])
def edit_fact(request: EditRequest):
    """Knowledge editing with gating mechanism (paper Eq. 1).
    Stores a KV pair: key=query_embedding, value=new_value.
    Future retrievals matching this embedding will return the edited value."""
    try:
        emb = get_embedding(request.query)
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            "INSERT INTO knowledge_edits (query_embedding, new_value) VALUES (?, ?)",
            (_pack_embedding(emb), request.new_value),
        )
        conn.commit()
        conn.close()
        return {"status": "edit_stored", "query": request.query}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/t2sql", dependencies=[Depends(verify_api_key)])
def text_to_sql(request: T2SQLRequest):
    """T2SQL Generator — translates natural language to SQL using few-shot CoT prompting."""
    try:
        db = request.db_path
        if not db or not os.path.exists(db):
            raise HTTPException(400, "Provide a valid db_path to an existing SQLite database.")

        schema = get_schema(db)
        if not schema:
            raise HTTPException(400, "Could not extract schema from database.")

        result = generate_sql(request.question, schema)
        sql = result.get("sql", "")
        thought = result.get("thought", "")

        # Execute the generated SQL
        conn = sqlite3.connect(db)
        conn.row_factory = sqlite3.Row
        try:
            cur = conn.execute(sql)
            rows = [dict(r) for r in cur.fetchall()]
        except Exception as sql_err:
            conn.close()
            return {"thought": thought, "sql": sql, "error": str(sql_err), "rows": []}
        conn.close()

        return {"thought": thought, "sql": sql, "rows": rows, "count": len(rows)}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
