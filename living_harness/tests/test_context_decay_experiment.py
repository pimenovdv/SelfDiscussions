import pytest
import tempfile
import os
import time
from living_harness.memory.vector_store import VectorStore

def test_context_decay_experiment():
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)

    try:
        store = VectorStore(db_path=path)

        # Add primary metadata rules
        store.add_item("Context A rule", [1.0, 0.0, 0.0], metadata={"is_abstract": True, "meta_level": 2, "last_used": time.time() - 4000})
        store.add_item("Context B rule", [0.0, 1.0, 0.0], metadata={"is_abstract": True, "meta_level": 2, "last_used": time.time()})

        meta_rules = [item for item in store.memory if item.get("metadata", {}).get("is_abstract", False)]
        assert len(meta_rules) == 2

        assert meta_rules[0]["metadata"]["meta_level"] == 2
        assert meta_rules[1]["metadata"]["meta_level"] == 2

        store.apply_decay(last_used_threshold=3600)

        meta_rules_after = [item for item in store.memory if item.get("metadata", {}).get("is_abstract", False)]
        assert len(meta_rules_after) == 2

        assert meta_rules_after[0]["metadata"]["meta_level"] == 1 # Decayed
        assert meta_rules_after[1]["metadata"]["meta_level"] == 2 # Did not decay

    finally:
        os.remove(path)
