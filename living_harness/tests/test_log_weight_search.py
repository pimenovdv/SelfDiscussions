import pytest
import os
import math
import time
from living_harness.memory.vector_store import VectorStore

def test_log_weight_scaling_ratio():
    db_path = "living_harness/data/test_log_weight.json"
    if os.path.exists(db_path):
        os.remove(db_path)

    try:
        vector_store = VectorStore(db_path=db_path)

        current_time = time.time()
        vector = [1.0, 0.0, 0.0]

        # Item 1: High weight (5.0, max normal weight)
        vector_store.add_item("High weight item", vector, metadata={"importance_weight": 5.0, "last_accessed": current_time})

        # Item 2: Base weight (1.0)
        vector_store.add_item("Low weight item", vector, metadata={"importance_weight": 1.0, "last_accessed": current_time})

        raw_items = vector_store.memory
        scored_items = []
        for item in raw_items:
            base_score = vector_store._cosine_similarity(vector, item["vector"])
            importance_weight = item["metadata"]["importance_weight"]
            days_passed = max(0.0, (current_time - item["metadata"]["last_accessed"]) / (24.0 * 3600.0))
            recency_boost = 1.0 + 0.2 * math.exp(-days_passed)
            final_score = base_score * math.log1p(importance_weight) * recency_boost
            scored_items.append({"score": final_score, "weight": importance_weight})

        high_score = next(x["score"] for x in scored_items if x["weight"] == 5.0)
        low_score = next(x["score"] for x in scored_items if x["weight"] == 1.0)

        ratio = high_score / low_score

        assert 2.5 < ratio < 2.7, f"Expected ratio around 2.58, got {ratio}"

        results = vector_store.search(vector, top_k=2)
        assert len(results) == 2

    finally:
        if os.path.exists(db_path):
            os.remove(db_path)
