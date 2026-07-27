import json
import random
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

COLLECTION_NAME = "scifact_minilm"
CORPUS_PATH = "data/scifact/corpus.jsonl"
QUERIES_PATH = "data/scifact/queries.jsonl"
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
    # Number of queries to test
    K = 5

    with open(QUERIES_PATH, "r") as f:
        all_queries = [json.loads(line) for line in f]

    sampled_queries = random.sample(all_queries, K)

    for query in sampled_queries:
        query_text = query["text"]
        results = search(query_text)

        result = results[0]
        print()
        print("Query: ", query_text)
        print("Point ID: ", result.id, "\tResult Score: ", result.score)
        print("Title: ", result.payload["title"][:150])  # type: ignore
        print("Text: ", result.payload["text"][:150])  # type: ignore
