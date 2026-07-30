import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer
import os


INDEX_PATH = "data/scifact.index"
DOC_IDS_PATH = "data/doc_ids.json"

# Assert that the index must be loaded and stored first
assert os.path.exists(INDEX_PATH), "Run faiss_index.py first"
assert os.path.exists(DOC_IDS_PATH), "Run faiss_index.py first"

# Model for embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")

# 1 Load the FAISS index from disk
index = faiss.read_index(INDEX_PATH)

# 2 Load the doc_ids.json
with open(DOC_IDS_PATH, "r") as file:
    doc_ids = json.load(file)


def search(query_text, k=5):
    # 3 Encode the query string using the model
    query_vector = model.encode(query_text).reshape(1, -1).astype(np.float32)

    # 4 Normalize the query
    faiss.normalize_L2(query_vector)

    # 5 Search using FAISS and return
    distances, indexes = index.search(query_vector, k)

    return [(doc_ids[i], float(s)) for i, s in zip(indexes[0], distances[0])]
