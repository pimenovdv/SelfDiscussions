import pytest
from living_harness.core.garbage_collection import ColdStorageGC
from living_harness.memory.vector_store import VectorStore
from living_harness.memory.partitioning import PartitionManager
import os

def test_importance_weight_accumulation():
    db_path = "living_harness/data/test_importance_weight.json"
    if os.path.exists(db_path):
        os.remove(db_path)
    vector_store = VectorStore(db_path=db_path)
    partition_manager = PartitionManager()

    gc = ColdStorageGC(vector_store, partition_manager)

    mem1 = {
        "text": "Pattern A",
        "vector": [1.0, 0.0, 0.0],
        "metadata": {
            "importance_weight": 1.0,
            "timestamp": 0
        }
    }
    mem2 = {
        "text": "Pattern A duplicate",
        "vector": [1.0, 0.0, 0.0],
        "metadata": {
            "importance_weight": 1.0,
            "timestamp": 0
        }
    }
    mem3 = {
        "text": "Pattern A duplicate 2",
        "vector": [1.0, 0.0, 0.0],
        "metadata": {
            "importance_weight": 2.5,
            "timestamp": 0
        }
    }
    mem4 = {
        "text": "Pattern B",
        "vector": [0.0, 1.0, 0.0],
        "metadata": {
            "importance_weight": 1.0,
            "timestamp": 0
        }
    }

    cold_memories = [mem1, mem2, mem3, mem4]

    defragmented = gc.compress_and_archive(cold_memories, current_storage_size=10000)

    assert len(defragmented) == 2, f"Expected 2 defragmented memories, got {len(defragmented)}"

    merged_mem_a = next(m for m in defragmented if m["vector"] == [1.0, 0.0, 0.0])
    merged_mem_b = next(m for m in defragmented if m["vector"] == [0.0, 1.0, 0.0])

    assert merged_mem_a["metadata"]["importance_weight"] == 4.5, f"Expected weight 4.5, got {merged_mem_a['metadata']['importance_weight']}"

    assert merged_mem_b["metadata"]["importance_weight"] == 1.0, f"Expected weight 1.0, got {merged_mem_b['metadata']['importance_weight']}"
    if os.path.exists(db_path):
        os.remove(db_path)
