import pytest
from living_harness.core.garbage_collection import ColdStorageGC
from living_harness.memory.vector_store import VectorStore
from living_harness.memory.partitioning import PartitionManager

def test_adaptive_threshold():
    vs = VectorStore(db_path="living_harness/data/test_vector_memory.json")
    pm = PartitionManager()
    gc = ColdStorageGC(vs, pm)

    # Check baseline
    assert abs(gc.get_adaptive_threshold(0) - 0.95) < 1e-5
    assert abs(gc.get_adaptive_threshold(5000) - 0.90) < 1e-5
    assert abs(gc.get_adaptive_threshold(10000) - 0.85) < 1e-5
    assert abs(gc.get_adaptive_threshold(20000) - 0.85) < 1e-5
