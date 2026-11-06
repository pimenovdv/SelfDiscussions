import pytest
import os
import json
from living_harness.memory.vector_store import VectorStore
from living_harness.memory.benchmarks import evaluate_semantic_retrieval

def test_semantic_benchmark(tmp_path):
    # Setup VectorStore
    db_path = str(tmp_path / "test_benchmark.json")
    vs = VectorStore(db_path=db_path)

    # Generate synthetic dataset of episodic memory
    for i in range(50):
        # A simple vector and text
        vector = [0.1 * i, 0.2, 0.3]
        vs.add_item(text=f"Episodic memory {i}", vector=vector, metadata={"type": "episodic", "id": str(i)}, domain="general")
        # vector_store searches return item dicts
        vs.memory[-1]["id"] = str(i) # Ensure ID exists for benchmark

    # Run benchmark BEFORE consolidation
    queries = [[0.1 * i, 0.2, 0.3] for i in range(10)]
    expected_ids = [[str(i)] for i in range(10)]

    metrics_before = evaluate_semantic_retrieval(vs, queries, expected_ids, top_k=5)

    # Perform consolidation
    vs.consolidate_abstractions(max_depth=3, use_cosine=True)

    # Since consolidation adds new concepts, we should also track if metric changes or not.
    # The benchmark simply expects IDs to match
    metrics_after = evaluate_semantic_retrieval(vs, queries, expected_ids, top_k=5)

    # Store logs to data folder
    log_data = {
        "metrics_before": metrics_before,
        "metrics_after": metrics_after
    }

    os.makedirs("living_harness/data", exist_ok=True)
    with open("living_harness/data/semantic_benchmark_logs.json", "w") as f:
        json.dump(log_data, f)

    assert "ndcg_at_k" in metrics_before
    assert "recall_at_k" in metrics_before
    assert "latency_sec_per_query" in metrics_before
    assert metrics_after["recall_at_k"] >= 0.0
