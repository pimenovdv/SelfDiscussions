import pytest
import math
from living_harness.memory.vector_store import VectorStore

@pytest.fixture
def store(tmp_path):
    store_path = tmp_path / "test_reflection_store.json"
    return VectorStore(db_path=str(store_path))

def test_reflect_on_monologue(store):
    store.add_item(
        text="I should focus on optimizing memory.",
        vector=[1.0, 0.0, 0.0],
        metadata={"type": "internal_monologue"},
        domain="abstract"
    )
    store.add_item(
        text="Memory optimization is my priority.",
        vector=[0.9, 0.1, 0.0],
        metadata={"type": "internal_monologue"},
        domain="abstract"
    )
    store.add_item(
        text="I wonder what time it is.",
        vector=[0.0, 1.0, 0.0],
        metadata={"type": "internal_monologue"},
        domain="abstract"
    )

    store.reflect_on_monologue(similarity_threshold=0.8)

    assert len(store.memory) == 2

    meta = [m for m in store.memory if m.get("metadata", {}).get("type") == "meta_knowledge"]
    assert len(meta) == 1
    assert "Memory optimization" in meta[0]["text"]
    assert meta[0]["metadata"]["importance_weight"] == 2.0

    expected_vec = [0.95, 0.05, 0.0]
    for actual, expected in zip(meta[0]["vector"], expected_vec):
        assert math.isclose(actual, expected, rel_tol=1e-5)
