import time
import pytest
from living_harness.core.garbage_collection import ColdStorageGC
from living_harness.memory.vector_store import VectorStore
from living_harness.memory.partitioning import PartitionManager

def test_apply_weight_decay(tmp_path):
    db_path = tmp_path / "test_memory_decay.json"
    vs = VectorStore(db_path=str(db_path))
    pm = PartitionManager()
    gc = ColdStorageGC(vector_store=vs, partition_manager=pm)

    vs.add_item("doc1", [1.0, 0.0], {"importance_weight": 5.0})
    vs.add_item("doc2", [0.0, 1.0], {"importance_weight": 1.2})
    vs.add_item("doc3", [0.5, 0.5], {})  # No weight

    gc.apply_weight_decay(decay_factor=0.8)

    assert vs.memory[0]["metadata"]["importance_weight"] == 4.0
    assert vs.memory[1]["metadata"]["importance_weight"] == 1.0  # Clipped to 1.0
    assert "importance_weight" not in vs.memory[2].get("metadata", {})

def test_gc_cycle_memory_freeing(tmp_path, monkeypatch):
    db_path = tmp_path / "test_memory_free.json"
    vs = VectorStore(db_path=str(db_path))
    pm = PartitionManager(hot_window_seconds=0.1)
    gc = ColdStorageGC(vector_store=vs, partition_manager=pm)

    # Mock autoencoder to return same vector
    monkeypatch.setattr(gc.autoencoder, "detect_and_correct", lambda v: v)

    # Add similar vectors to trigger deduplication
    vs.add_item("doc1", [1.0, 0.0], {"importance_weight": 2.0, "timestamp": time.time() - 1.0})
    vs.add_item("doc2", [1.0, 0.01], {"importance_weight": 3.0, "timestamp": time.time() - 1.0})

    assert len(vs.memory) == 2

    # Run GC
    gc.gc_cycle()

    # Memory should be compressed (1 deduplicated item)
    assert len(vs.memory) == 1

    # The weight should be the sum of decayed weights
    # Original: doc1 (2.0), doc2 (3.0). Decayed by 0.99: 1.98, 2.97. Sum: 4.95
    weight = vs.memory[0]["metadata"]["importance_weight"]
    assert abs(weight - 4.95) < 0.001
