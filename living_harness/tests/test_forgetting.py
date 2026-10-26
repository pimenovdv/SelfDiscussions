import pytest
from ..memory.vector_store import VectorStore
import os

def test_forgetting_mechanism(tmp_path):
    store = VectorStore(db_path=str(tmp_path / "test_memory.json"))
    store.add_item("memory 1", [0.1, 0.2], metadata={"importance_weight": 0.4})
    store.add_item("memory 2", [0.2, 0.3], metadata={"importance_weight": 0.8})
    store.add_item("abstract memory", [0.3, 0.4], metadata={"importance_weight": 0.2, "is_abstract": True})
    store.add_item("meta memory", [0.4, 0.5], metadata={"importance_weight": 0.3, "type": "meta_knowledge"})

    assert len(store.memory) == 4

    store.forget_old_memories(threshold=0.5)

    assert len(store.memory) == 3
    assert store.memory[0].get("text") == "memory 2"
    assert store.memory[1].get("text") == "abstract memory"
    assert store.memory[2].get("text") == "meta memory"
