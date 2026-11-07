import pytest
import os
import tempfile
import json
from living_harness.memory.vector_store import VectorStore

def test_predictive_modulation_pulls_vector():
    fd, temp_path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        vs = VectorStore(db_path=temp_path)
        # Create a meta-rule Level 2
        vs.add_item(
            text="Meta Rule: Active Modulation",
            vector=[1.0, 1.0, 1.0, 1.0],
            metadata={"is_abstract": True, "meta_level": 2}
        )

        input_vector = [0.1, 0.1, 0.1, 0.1]
        modulated = vs.predictive_modulation(input_vector, intensity=0.5)

        # Check if vector is modulated towards [1.0, 1.0, 1.0, 1.0]
        assert modulated != input_vector
        for val in modulated:
            assert val > 0.1

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
