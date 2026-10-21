import pytest
from living_harness.memory.vector_store import VectorStore
import os

def test_dynamic_quantize():
    db_path = "data/test_dynamic_precision.json"
    store = VectorStore(db_path=db_path)
    store.clear()

    v1 = [1.0, 0.0, 0.0]
    v2 = [0.0, 1.0, 0.0]

    store.add_item("important", v1, metadata={"importance_weight": 2.5}, quantize=False)
    store.add_item("unimportant", v2, metadata={"importance_weight": 1.1}, quantize=False)

    # Both should be float initially
    assert store.memory[0]["metadata"].get("is_quantized", False) == False
    assert store.memory[1]["metadata"].get("is_quantized", False) == False

    # Apply dynamic quantize
    store.dynamic_quantize(threshold=2.0)

    # Important should stay float, unimportant should be quantized
    assert store.memory[0]["metadata"].get("is_quantized", False) == False
    assert store.memory[1]["metadata"].get("is_quantized", False) == True

    # Search should still work
    res1 = store.search(v1)
    assert res1[0]["text"] == "important"

    res2 = store.search(v2)
    assert res2[0]["text"] == "unimportant"

    store.clear()
