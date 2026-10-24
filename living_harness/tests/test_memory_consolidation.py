import pytest
import os
import json
import math
from living_harness.memory.vector_store import VectorStore

@pytest.fixture
def store(tmp_path):
    store_path = tmp_path / "test_consolidation.json"
    return VectorStore(db_path=str(store_path))

def test_memory_consolidation(store):
    # Add two similar abstract concepts
    store.add_item(
        text="Concept A",
        vector=[1.0, 0.1, 0.0],
        metadata={"is_abstract": True, "importance_weight": 2.0},
        domain="abstract"
    )
    store.add_item(
        text="Concept B",
        vector=[0.9, 0.2, 0.0],
        metadata={"is_abstract": True, "importance_weight": 3.0},
        domain="abstract"
    )

    # Add a different concept
    store.add_item(
        text="Concept C",
        vector=[0.0, 1.0, 0.0],
        metadata={"is_abstract": True, "importance_weight": 1.5},
        domain="abstract"
    )

    # Run consolidation with a threshold that will merge A and B but not C
    store.consolidate_memory(similarity_threshold=0.8)

    # Verify the results
    assert len(store.memory) == 2

    consolidated = [item for item in store.memory if item.get("text").startswith("Consolidated Concept")]
    assert len(consolidated) == 1

    merged_item = consolidated[0]
    assert merged_item["metadata"]["importance_weight"] == 5.0
    assert "Concept A" in merged_item["metadata"]["consolidated_from"]
    assert "Concept B" in merged_item["metadata"]["consolidated_from"]

    expected_vector = [0.95, 0.15, 0.0]
    for actual, expected in zip(merged_item["vector"], expected_vector):
        assert math.isclose(actual, expected, rel_tol=1e-5)

    # Check that Concept C is untouched
    concept_c = [item for item in store.memory if item.get("text") == "Concept C"]
    assert len(concept_c) == 1
    assert concept_c[0]["metadata"]["importance_weight"] == 1.5
