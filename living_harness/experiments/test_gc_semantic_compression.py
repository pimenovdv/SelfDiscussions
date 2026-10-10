import os
import time
from living_harness.core.garbage_collection import ColdStorageGC
from living_harness.memory.vector_store import VectorStore
from living_harness.memory.partitioning import PartitionManager

def test_gc_semantic_compression():
    test_db = "living_harness/data/test_gc_compression_db.json"
    if os.path.exists(test_db):
        os.remove(test_db)

    vector_store = VectorStore(db_path=test_db)
    partition_manager = PartitionManager(hot_window_seconds=1)

    gc = ColdStorageGC(vector_store, partition_manager)

    # Create two almost identical vectors (duplicates)
    vec1 = [0.5] * 256
    vec2 = [0.51] * 256
    vec3 = [-0.2] * 256 # Different vector

    vector_store.add_item("memory 1", vec1, {"timestamp": time.time() - 2})
    vector_store.add_item("memory 2", vec2, {"timestamp": time.time() - 2})
    vector_store.add_item("memory 3", vec3, {"timestamp": time.time() - 2})

    assert len(vector_store.memory) == 3

    gc.gc_cycle()

    # One of the duplicates should be removed, leaving 2
    assert len(vector_store.memory) == 2

    if os.path.exists(test_db):
        os.remove(test_db)

if __name__ == "__main__":
    test_gc_semantic_compression()
