import pytest
from typing import List
from living_harness.memory.vector_store import VectorStore

def test_emotional_resonance():
    store = VectorStore(db_path="data/test_emotion.json")
    store.clear()

    vec1 = [0.1, 0.2, 0.3]
    vec2 = [0.1, 0.2, 0.3] # Identical vector to test pure emotional boost

    store.add_item("Happy memory", vec1, emotion="happy")
    store.add_item("Sad memory", vec2, emotion="sad")

    # Query with identical vector, but happy emotion
    results = store.search(vec1, top_k=2, emotional_state="happy")

    assert len(results) == 2
    assert results[0]["text"] == "Happy memory"
    assert results[1]["text"] == "Sad memory"

    # Query with identical vector, but sad emotion
    results = store.search(vec1, top_k=2, emotional_state="sad")

    assert results[0]["text"] == "Sad memory"
    assert results[1]["text"] == "Happy memory"
