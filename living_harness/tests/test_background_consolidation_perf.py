import pytest
import time
from living_harness.memory.vector_store import VectorStore
from living_harness.memory.consolidation import MemoryConsolidator

def test_background_consolidation_performance(tmp_path):
    db_path = tmp_path / "test_perf_memory.json"
    store = VectorStore(db_path=str(db_path))
    consolidator = MemoryConsolidator(vector_store=store)

    # Add items
    for i in range(100):
        store.add_item(f"item {i}", [float(i), 0.0], {"type": "episodic", "importance_weight": 2.5, "last_accessed": time.time() - 48*3600})

    # Measure search time before
    start_time = time.time()
    res1 = store.search([50.0, 0.0])
    search_time_before = time.time() - start_time

    # Run sleep cycle
    start_time = time.time()
    consolidator.run_sleep_cycle()
    sleep_cycle_time = time.time() - start_time

    # Measure search time after
    start_time = time.time()
    res2 = store.search([50.0, 0.0])
    search_time_after = time.time() - start_time

    assert sleep_cycle_time < 5.0  # Ensure it's not too slow
    assert search_time_after < 0.1 # Ensure search is still fast
