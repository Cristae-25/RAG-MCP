import json
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

COLLECTION_NAME = "scifact_minilm"
CORPUS_PATH = "data/scifact/corpus.jsonl"
UPSERT_BATCH_SIZE = 256

client = QdrantClient(url="http://localhost:6333")
model = SentenceTransformer("all-MiniLM-L6-v2")


def search(query_text, k=5):
    query_vector = model.encode(query_text).tolist()

    results = client.query_points(
        collection_name=COLLECTION_NAME, query=query_vector, limit=k
    )

    return results.points


if __name__ == "__main__":
    query = input("Search Query: ")
    results = search(query)

    for result in results:
        print("Point ID: ", result.id, "\tResult Score: ", result.score)
        print("Title: ", result.payload["title"][:150])  # type: ignore
        print("Text: ", result.payload["text"][:150])  # type: ignore
        print()
