# RAG Evaluation + MCP Server — 4-Week Sprint Tracker

**Project:** Qdrant/FAISS Retrieval Evaluation + MCP Server, tested via Antigravity CLI (free Gemini)
**Cadence:** 2-3 days/week, ~2-3 hr sessions, alongside internship
**Goal:** A portfolio repo + quantified metrics for resume bullets

---

## How to use this tracker
- Each week has **Day-sized chunks** (not hour-sized) so you can pick them up/put them down around internship work.
- Each day ends with a **"Proof of Life"** — something you can literally see (output, screenshot, chart) that proves the work is real.
- Each week ends with a **Metrics Checkpoint** — fill in the blanks. These become your resume bullets in Week 4.
- Update the ✅ boxes as you go. If a day slips, just move it — don't restart the sprint.

---

## WEEK 1 — Foundation: Corpus + Vector Store

### Day 1 (~2-3 hrs)
- [X] Install Docker, pull/run Qdrant locally (`docker run -p 6333:6333 qdrant/qdrant`)
- [X] Install `faiss-cpu`, `beir`, `sentence-transformers`
- [X] Download BEIR **SciFact** dataset (~5K docs — small enough to run fast, big enough to be credible)
- **Proof of Life:** Qdrant dashboard loads at `localhost:6333/dashboard`; SciFact corpus unzipped and readable in Python.

### Day 2 (~2-3 hrs)
- [X] Pick embedding model (recommend `all-MiniLM-L6-v2` — free, local, fast, 384-dim)
- [X] Embed the full SciFact corpus, load vectors into Qdrant
- **Proof of Life:** Run `qdrant_client.count(collection_name)` → prints doc count matching corpus size.

### Day 3 (optional, if 3rd day available)
- [ ] Write a basic `search(query, k=5)` function
- [ ] Run 3-5 manual test queries, eyeball whether top results look relevant
- **Proof of Life:** Terminal output showing a real query → ranked list of real document titles + similarity scores.

### 📊 Week 1 Metrics Checkpoint
| Metric | Value |
|---|---|
| Corpus used | BEIR SciFact|
| # documents indexed |5,183 |
| Embedding model + dimension | all-MiniLM-L6-v2, 384 dimensions|
| Indexing time (total) | 17 seconds|
| Vector store used | Qdrant |

Lessons Learned:
Qdrant has a 32MB request payload limit so I had to batch using 256

---

## WEEK 2 — Evaluation Harness (Qdrant vs. FAISS)

### Day 1 (~2-3 hrs)
- [ ] Load BEIR's qrels (ground-truth relevance judgments) for SciFact
- [ ] Wire up BEIR's `EvaluateRetrieval` to score your Qdrant results: nDCG@10, Recall@10, MAP@10, Precision@10
- **Proof of Life:** A printed metrics dict, e.g. `{'NDCG@10': 0.71, 'Recall@10': 0.83, ...}`

### Day 2 (~2-3 hrs)
- [ ] Build a second index using raw FAISS (flat or IVF) on the same embeddings
- [ ] Run the **same** eval script against the FAISS index
- **Proof of Life:** Side-by-side table — Qdrant metrics vs. FAISS metrics on identical data.

### Day 3 (~2-3 hrs)
- [ ] Add latency benchmarking: run 50-100 queries, log per-query time, compute **mean** and **p95**
- [ ] Plot a simple bar chart (Qdrant vs FAISS: accuracy metrics + latency) — matplotlib is fine
- **Proof of Life:** A saved `.png` chart — this becomes your first portfolio visual.

### 📊 Week 2 Metrics Checkpoint
| Metric | Qdrant | FAISS |
|---|---|---|
| nDCG@10 | | |
| Recall@10 | | |
| MRR | | |
| Mean latency (ms) | | |
| p95 latency (ms) | | |

---

## WEEK 3 — MCP Server

### Day 1 (~2-3 hrs)
- [ ] Install MCP Python SDK — **pin to v1.x** (v2 is pre-release, don't use it)
- [ ] Scaffold a server exposing one tool: `search_documents(query: str, k: int)` → wraps your Week 1-2 search function
- **Proof of Life:** MCP Inspector connects to your server and lists the tool with correct schema.

### Day 2 (~2-3 hrs)
- [ ] Add a second tool: `get_eval_metrics()` → returns your Week 2 metrics as structured JSON
- [ ] Test both tools via MCP Inspector, confirm real responses (not stubs)
- **Proof of Life:** Inspector shows a live call to `search_documents("query")` returning real SciFact chunks.

### Day 3 (~2-3 hrs)
- [ ] Connect your MCP server to **Claude Desktop** (config in `claude_desktop_config.json`)
- [ ] Ask Claude a natural-language question that forces it to invoke your tool (e.g., "search my document server for X")
- **Proof of Life:** Screenshot of Claude Desktop autonomously calling your tool mid-conversation.

### 📊 Week 3 Metrics Checkpoint
| Metric | Value |
|---|---|
| # tools exposed | |
| Tool call success rate (manual tests) | |
| Round-trip latency (query → MCP → response) | |
| Client used to test | Claude Desktop |

---

## WEEK 4 — Gemini Agent Test + Polish

### Day 1 (~2-3 hrs)
- [ ] Install **Antigravity CLI** (`agy`) — free successor to Gemini CLI, native MCP support
  - `curl -fsSL https://antigravity.google/cli/install.sh | bash`
- [ ] Register your MCP server with `agy`, verify via `/mcp list` (or `agy` equivalent)
- **Fallback path if `agy` gives you trouble:** write a small Python script using the `google-generativeai` SDK with manual function-calling that forwards calls to your MCP server's tools — more code, but shows you understand the client-side glue, not just the CLI's built-in support.
- **Proof of Life:** `agy` recognizes and lists your MCP tools.

### Day 2 (~2-3 hrs)
- [ ] Run 3-5 natural-language prompts through `agy` that require it to invoke `search_documents` and/or `get_eval_metrics`
- [ ] Capture terminal logs showing autonomous tool invocation (not you calling it manually)
- **Proof of Life:** Saved terminal transcript / screen recording of Gemini agent autonomously using your MCP tools.

### Day 3 (~2-3 hrs)
- [ ] Write README (architecture diagram, setup instructions, metrics table)
- [ ] Record a 60-90 sec demo GIF/video: query → MCP tool call → Gemini agent response
- [ ] Compile final resume bullets (see below)
- **Proof of Life:** Public GitHub repo, README renders correctly, demo asset attached.

### 📊 Week 4 Metrics Checkpoint
| Metric | Value |
|---|---|
| Agent client used | Antigravity CLI (or fallback) |
| # successful autonomous tool calls | |
| Any failure modes observed | |

---

## 🎯 Final Resume Bullets (fill in once complete)

Draft template — replace bracketed values with your real numbers from the checkpoints above:

> Built a retrieval-augmented search system indexing **[N] documents** using Qdrant (HNSW) and FAISS, achieving **[X]% Recall@10** and **[Y]% nDCG@10**, benchmarked against the BEIR SciFact IR standard.

> Designed and deployed an MCP (Model Context Protocol) server exposing search and evaluation tools, enabling autonomous tool invocation by both Claude and Gemini-based AI agents with **[Z]ms** round-trip latency.

---

## Weekly Check-in Prompt (for our conversations)
At the end of each week, tell me:
1. Which days you completed
2. The numbers from that week's Metrics Checkpoint
3. Anything that blocked you

I'll help you troubleshoot, adjust the next week's scope if needed, and keep this tracker updated.
