import json

from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval

from search import search
from faiss_search import search as f_search

DATA_PATH = "data/scifact"
K_VALUES = [10]
METRICS_OUT_PATH = "results/day2_eval_metrics.json"


def build_results(
    search_fn, queries: dict[str, str], k: int
) -> dict[str, dict[str, float]]:
    results = {}
    for qid, text in queries.items():
        hits = search_fn(text, k=k)
        results[qid] = {doc_id: score for doc_id, score in hits}  # type: ignore
    return results


def qdrant_search_adapter(query_text, k):
    hits = search(query_text, k=k)
    return [(hit.payload["doc_id"], hit.score) for hit in hits]


def main():
    backends = {
        "qdrant": qdrant_search_adapter,
        "faiss": f_search,
    }
    _corpus, queries, qrels = GenericDataLoader(data_folder=DATA_PATH).load(
        split="test"
    )
    print(f"Loaded {len(queries)} test queries and qrels.")

    all_metrics = {}

    for name, search_fn in backends.items():
        results = build_results(search_fn, queries, k=max(K_VALUES))
        ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(
            qrels, results, K_VALUES
        )
        all_metrics[name] = {**ndcg, **_map, **recall, **precision}
        print(f"{name}: {all_metrics[name]}")

    with open(METRICS_OUT_PATH, "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"Saved metrics to {METRICS_OUT_PATH}")


if __name__ == "__main__":
    main()
