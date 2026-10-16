import pytest
import time
from living_harness.memory.vector_store import VectorStore
from living_harness.core.garbage_collection import ColdStorageGC
from living_harness.memory.partitioning import PartitionManager

def test_time_based_decay(tmp_path):
    db_path = tmp_path / "test_memory.json"
    store = VectorStore(db_path=str(db_path))
    manager = PartitionManager()
    gc = ColdStorageGC(vector_store=store, partition_manager=manager)

    current_time = time.time()
    one_day = 24 * 3600

    # Add items with last_accessed in the past
    store.add_item("Recent item", [0.1, 0.2], metadata={"importance_weight": 5.0, "last_accessed": current_time})
    store.add_item("Old item", [0.3, 0.4], metadata={"importance_weight": 5.0, "last_accessed": current_time - 10 * one_day})

    # Run apply_weight_decay and simulate current_time passing
    gc.apply_weight_decay(time_based=True, decay_rate_per_day=0.05)

    recent = store.memory[0]["metadata"]["importance_weight"]
    old = store.memory[1]["metadata"]["importance_weight"]

    assert pytest.approx(recent, 0.001) == 5.0 # No time passed for this one since decay calculation uses last_accessed if no last_decayed
    assert old < 5.0
    assert pytest.approx(old, 0.001) == max(1.0, 5.0 * (1.0 - 0.05 * 10))
