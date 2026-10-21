import pytest
from living_harness.memory.vector_store import VectorStore
import os
import math

def test_vector_quantization():
    db_path = "data/test_vector_quantization.json"
    store = VectorStore(db_path=db_path)
    store.clear()

    # 1. Edge Case: min_v == max_v
    zero_vec = [0.0, 0.0, 0.0]
    store.add_item("zeros", zero_vec, domain="test", quantize=True)
    assert store.memory[0]["vector"] == [0, 0, 0]
    assert store.memory[0]["metadata"]["is_quantized"] == True

    # Check that decoding gives back original
    res = store.search(zero_vec, domain="test")
    assert res[0]["text"] == "zeros"

    store.clear()

    # 2. General case
    orig_vec = [0.1, 0.5, 0.9, -0.3]
    store.add_item("gen", orig_vec, domain="test", quantize=True)

    # should be ints
    quant_vec = store.memory[0]["vector"]
    assert all(isinstance(v, int) for v in quant_vec)
    assert min(quant_vec) >= 0
    assert max(quant_vec) <= 255

    # search with original should still return it as closest
    res = store.search(orig_vec, domain="test")
    assert res[0]["text"] == "gen"

    # 3. Check search quality and distance (quantized should have high cosine similarity with original)
    decoded = store._get_float_vector(store.memory[0])
    sim = store._cosine_similarity(orig_vec, decoded)
    assert sim > 0.98 # high similarity

    store.clear()

def test_vector_mixed_compatibility():
    db_path = "data/test_mixed_comp.json"
    store = VectorStore(db_path=db_path)
    store.clear()

    v1 = [1.0, 0.5, 0.0]
    v2 = [0.0, 0.5, 1.0]

    # One quantized, one not
    store.add_item("quantized", v1, quantize=True)
    store.add_item("normal", v2, quantize=False)

    assert store.memory[0]["metadata"].get("is_quantized", False) == True
    assert store.memory[1]["metadata"].get("is_quantized", False) == False

    # Search should work and transparently compare both
    res = store.search([1.0, 0.4, 0.0])
    assert res[0]["text"] == "quantized"

    res2 = store.search([0.0, 0.4, 1.0])
    assert res2[0]["text"] == "normal"

    store.clear()
