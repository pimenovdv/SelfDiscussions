import os
import json
from living_harness.memory.vector_store import VectorStore

def test_internal_monologue_type():
    db_path = "test_memory.json"
    if os.path.exists(db_path):
        os.remove(db_path)
    vs = VectorStore(db_path=db_path)
    vs.add_item("My inner thought", [0.1, 0.2], {"type": "internal_monologue"})
    assert len(vs.memory) == 1
    assert vs.memory[0]["metadata"]["type"] == "internal_monologue"
    if os.path.exists(db_path):
        os.remove(db_path)
