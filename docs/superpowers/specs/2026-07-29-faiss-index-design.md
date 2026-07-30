# FAISS Second Index Design
**Date:** 2026-07-29
**Sprint:** Week 2 Day 2
**Goal:** Build a second retrieval index using raw FAISS alongside the existing Qdrant index, run the same BEIR evaluation harness against both, and produce a side-by-side metrics table.

---

## Context

The project benchmarks Qdrant (HNSW) vs. FAISS (exact search) on the SciFact corpus (5,183 docs, `all-MiniLM-L6-v2` embeddings, 384 dimensions, cosine similarity). Week 2 Day 2 produces the comparison table that becomes the core resume metric. Week 2 Day 3 adds latency benchmarking on top of this foundation. Week 3's MCP server exposes the results via `get_eval_metrics()`.

---

## Index Type

**`faiss.IndexFlatIP`** (exact inner product search).

Vectors are L2-normalized before indexing, making inner product equal to cosine similarity. At 5,183 documents, brute-force search runs in <10ms per query — the speed difference vs. IVF is negligible. Exact search isolates the index algorithm as the only variable between Qdrant and FAISS, producing a clean experimental comparison.

---

## Architecture & Files

```
src/
  embed_and_index.py   ← MODIFIED: save embeddings.npy + doc_ids.json after Qdrant upsert
  faiss_index.py       ← NEW: load artifacts, build IndexFlatIP, save scifact.index
  faiss_search.py      ← NEW: load index + doc_ids, expose search(query_text, k)
  search.py            ← UNCHANGED (Qdrant search)
  evaluate.py          ← MODIFIED: accept {name: search_fn} dict, run all, side-by-side table

data/
  embeddings.npy       ← NEW (~7.6 MB, float32, shape 5183×384)
  doc_ids.json         ← NEW (~150 KB, list where index == FAISS integer ID)
  scifact.index        ← NEW (~7.6 MB, serialized IndexFlatIP)

results/
  day1_eval_metrics.json   ← UNCHANGED
  day2_eval_metrics.json   ← NEW: {"qdrant": {...}, "faiss": {...}}
```

**Run order (one-time setup):**
1. `embed_and_index.py` — embeds corpus, loads Qdrant, saves `.npy` + `doc_ids.json`
2. `faiss_index.py` — loads `.npy`, builds FAISS index, saves `scifact.index`

**Run anytime:**
3. `evaluate.py` — loads both backends, runs BEIR eval, prints table, saves `day2_eval_metrics.json`

---

## Data Flow

### `embed_and_index.py` (3 lines added at end)
After the existing Qdrant upsert, append:
```python
np.save("data/embeddings.npy", embeddings)
with open("data/doc_ids.json", "w") as f:
    json.dump([d["_id"] for d in docs], f)
```
`embeddings` is already in scope as a numpy float32 array from `model.encode()`.

### `faiss_index.py` (new)
1. Load `data/embeddings.npy` → numpy float32 array (5183, 384)
2. L2-normalize all vectors with `faiss.normalize_L2(embeddings)`
3. Create `faiss.IndexFlatIP(384)`, add all vectors
4. `faiss.write_index(index, "data/scifact.index")`

### `faiss_search.py` (new)
- Load `data/scifact.index`, `data/doc_ids.json`, and `SentenceTransformer("all-MiniLM-L6-v2")` once at module level
- `search(query_text, k)`:
  1. Encode query with the module-level model
  2. Reshape to `(1, 384)`, L2-normalize with `faiss.normalize_L2()`
  3. `index.search(vec, k)` → returns `(scores, indices)`
  4. Return `[(doc_ids[i], float(s)) for i, s in zip(indices[0], scores[0])]`

### `evaluate.py` (refactored)
Both search functions must return the same type: `list[tuple[str, float]]` — `(doc_id, score)` pairs. The current Qdrant `search()` returns `list[ScoredPoint]`, so `evaluate.py` wraps it with a thin adapter:

```python
def qdrant_search_adapter(query_text: str, k: int) -> list[tuple[str, float]]:
    hits = qdrant_search(query_text, k=k)
    return [(hit.payload["doc_id"], hit.score) for hit in hits]

backends = {
    "qdrant": qdrant_search_adapter,
    "faiss":  faiss_search,
}
```

`search.py` stays unchanged. The adapter lives only in `evaluate.py`.

For each backend: `build_results()` → `EvaluateRetrieval.evaluate()` → collect into dict keyed by backend name. Print side-by-side table, write `results/day2_eval_metrics.json` in one pass.

BEIR expects `{qid: {doc_id: score}}`. Both backends return `(doc_id, score)` pairs via this interface, feeding identical downstream code.

---

## Error Handling

| Location | Failure | Guard |
|---|---|---|
| `faiss_search.py` module load | `scifact.index` missing | `assert os.path.exists(path), "Run faiss_index.py first"` |
| `faiss_search.py` module load | `doc_ids.json` missing | Same pattern |
| `faiss_index.py` | Normalization not applied at search time | Normalize explicitly in both `faiss_index.py` and `faiss_search.py` — never rely on model defaults |
| `evaluate.py` | FAISS scores are numpy float32, BEIR expects float | `float(score)` cast in result builder |

---

## Output Format

`results/day2_eval_metrics.json`:
```json
{
  "qdrant": {"NDCG@10": 0.64508, "MAP@10": 0.59593, "Recall@10": 0.78333, "P@10": 0.08833},
  "faiss":  {"NDCG@10": 0.xxxxx, "MAP@10": 0.xxxxx, "Recall@10": 0.xxxxx, "P@10": 0.xxxxx}
}
```

`day1_eval_metrics.json` is preserved unchanged as the Qdrant-only historical run.

Re-running `evaluate.py` overwrites `day2_eval_metrics.json` with identical values (evaluation is deterministic).

---

## Proof of Life

| Step | Check |
|---|---|
| After `embed_and_index.py` | `data/embeddings.npy` shape == (5183, 384); `len(doc_ids.json)` == 5183 |
| After `faiss_index.py` | `faiss.read_index("data/scifact.index").ntotal` == 5183 |
| After `evaluate.py` | Side-by-side table printed; FAISS metrics match Qdrant within floating-point rounding |

**Expected result:** `IndexFlatIP` on L2-normalized vectors is mathematically equivalent to cosine similarity exact search. FAISS metrics should be identical or within rounding of Qdrant's. A significant deviation signals a missing normalization step.

---

## Forward Compatibility

- **Week 2 Day 3 (latency):** `evaluate.py` already loops over backends — add `time.perf_counter()` wrappers around `build_results()` per backend.
- **Week 3 MCP `get_eval_metrics()`:** Read and return `results/day2_eval_metrics.json` directly.
- **Hybrid search (stretch goal):** A future `hybrid_search.py` could combine Qdrant + FAISS scores via RRF; register as a third backend in `evaluate.py`.
