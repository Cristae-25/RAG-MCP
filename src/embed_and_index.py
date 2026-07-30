import json
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

COLLECTION_NAME = "scifact_minilm"
CORPUS_PATH = "data/scifact/corpus.jsonl"
EMBEDDINGS_PATH = "data/embeddings.npy"
DOC_IDS_PATH = "data/doc_ids.json"
UPSERT_BATCH_SIZE = 256

client = QdrantClient(url="http://localhost:6333")
model = SentenceTransformer("all-MiniLM-L6-v2")
dim = model.get_embedding_dimension()
assert dim is not None, "Could not determine embedding dimension from model"

# recreate collection fresh each run (fine for a POC)
if client.collection_exists(COLLECTION_NAME):
    client.delete_collection(COLLECTION_NAME)

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
)

# Read each document from the jsonl
docs = []
with open(CORPUS_PATH, "r") as f:
    for line in f:
        docs.append(json.loads(line))

print(f"Loaded {len(docs)} documents. Embedding...")

# Save each document's title and contents
texts = [d.get("title", "") + " " + d.get("text", "") for d in docs]

# Create embeddings for each document
embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)

points = [
    PointStruct(
        id=i,
        vector=embeddings[i].tolist(),
        payload={
            "doc_id": docs[i]["_id"],
            "title": docs[i].get("title", ""),
            "text": docs[i].get("text", ""),
        },
    )
    for i in range(len(docs))
]

# Save the embeddings into Qdrant
print("Upserting into Qdrant...")
for start in range(0, len(points), UPSERT_BATCH_SIZE):
    batch = points[start : start + UPSERT_BATCH_SIZE]
    client.upsert(collection_name=COLLECTION_NAME, points=batch, wait=True)
    print(f"  upserted {start + len(batch)}/{len(points)}")

count = client.count(collection_name=COLLECTION_NAME).count
print(f"Done. Collection '{COLLECTION_NAME}' now has {count} vectors.")

np.save(EMBEDDINGS_PATH, embeddings)

with open(DOC_IDS_PATH, "w") as f:
    json.dump([d["_id"] for d in docs], f)
