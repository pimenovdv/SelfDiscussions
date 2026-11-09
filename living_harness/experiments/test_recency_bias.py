import json
import time
import tempfile
import os
from living_harness.memory.vector_store import VectorStore

def test_recency_bias_experiment():
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)

    try:
        store = VectorStore(db_path=path)
        store.clear()

        current = time.time()

        # 1 year ago, user stated a fact
        store.add_item("The user lives in New York.", [0.8, 0.2, 0.1], metadata={"timestamp": current - 31536000})

        # Yesterday, user moved
        store.add_item("The user just moved to London.", [0.75, 0.3, 0.1], metadata={"timestamp": current - 86400})

        # Query vector close to both
        query = [0.8, 0.25, 0.1]

        # Standard Vector Search (might return the older, more exact vector match)
        semantic_results = store.search(query, top_k=2)
        semantic_texts = [r["text"] for r in semantic_results]

        # Temporal retrieval (forces recency bias)
        # Get memory from the last week
        recent_results = store.retrieve_by_time(current - 604800, current)
        recent_texts = [r["text"] for r in recent_results]

        results = {
            "semantic_search": semantic_texts,
            "temporal_retrieval_1_week": recent_texts
        }

        # Ensure data directory exists
        os.makedirs("data", exist_ok=True)
        with open("data/recency_bias_metrics.json", "w") as f:
            json.dump(results, f, indent=4)

        print(json.dumps(results, indent=2))

    finally:
        os.remove(path)

if __name__ == "__main__":
    test_recency_bias_experiment()
