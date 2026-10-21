import pytest
import time
from living_harness.memory.vector_store import VectorStore
from living_harness.memory.consolidation import MemoryConsolidator

def test_background_consolidation(tmp_path):
    db_path = tmp_path / "test_bg_memory.json"
    store = VectorStore(db_path=str(db_path))
    consolidator = MemoryConsolidator(vector_store=store)

    store.add_item("item 1", [1.0, 0.0], {"type": "episodic", "importance_weight": 2.5, "last_accessed": time.time() - 48*3600}, quantize=False)
    store.add_item("item 3", [0.5, 0.5], {"importance_weight": 3.0, "last_accessed": time.time()}, quantize=False)

    assert len(store.memory) == 2

    consolidator.run_sleep_cycle()

    assert len(store.memory) == 2
    semantic_rules = [m for m in store.memory if m.get("metadata", {}).get("type") == "semantic"]
    assert len(semantic_rules) == 1
