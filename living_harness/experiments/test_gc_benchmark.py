import time
import os
from living_harness.memory.vector_store import VectorStore
from living_harness.memory.partitioning import PartitionManager
from living_harness.core.garbage_collection import ColdStorageGC

def test_gc_benchmark():
    # Setup
    store = VectorStore("test_benchmark_store.json")
    store.clear()
    partition = PartitionManager(hot_window_seconds=10)
    gc = ColdStorageGC(store, partition)

    # Add a lot of memories with vectors
    num_memories = 1000
    for i in range(num_memories):
        # some cold memories, some hot
        timestamp = time.time() - (20 if i % 2 == 0 else 5)
        store.add_item(f"memory_{i}", vector=[0.1] * 128, metadata={"timestamp": timestamp})

    start_time = time.time()
    gc.gc_cycle()
    end_time = time.time()

    duration = end_time - start_time
    print(f"GC cycle completed in {duration:.4f} seconds for {num_memories} items")

    # Cleanup
    if os.path.exists("test_benchmark_store.json"):
        os.remove("test_benchmark_store.json")

    assert duration < 5.0 # Just a basic threshold check
