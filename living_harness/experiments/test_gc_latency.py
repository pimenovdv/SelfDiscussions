import time
import os
import pytest
from living_harness.memory.vector_store import VectorStore
from living_harness.memory.partitioning import PartitionManager
from living_harness.core.garbage_collection import ColdStorageGC

def test_gc_latency():
    db_path = "living_harness/data/test_gc_latency.json"
    if os.path.exists(db_path):
        os.remove(db_path)

    vs = VectorStore(db_path)
    # create fragmented cold data
    for i in range(2000):
        vs.add_item(
            text=f"Frag text {i}",
            vector=[float(i), 1.0, 0.5],
            metadata={"timestamp": time.time() - 100000} # very cold
        )

    for i in range(200):
        vs.add_item(
            text=f"Hot text {i}",
            vector=[float(i), 0.5, 1.0],
            metadata={"timestamp": time.time()} # hot
        )

    query_vector = [10.0, 1.0, 0.5]

    # test search latency before GC
    start_time = time.time()
    results_before = vs.search(query_vector, top_k=5)
    latency_before = time.time() - start_time

    pm = PartitionManager(hot_window_seconds=86400.0)
    gc = ColdStorageGC(vs, pm)

    # Run GC
    gc.gc_cycle()

    # test search latency after GC
    start_time = time.time()
    results_after = vs.search(query_vector, top_k=5)
    latency_after = time.time() - start_time

    print(f"Latency before: {latency_before:.6f}, after: {latency_after:.6f}")
    assert latency_after <= latency_before or latency_after < 0.05

    if os.path.exists(db_path):
        os.remove(db_path)

if __name__ == "__main__":
    test_gc_latency()
