import os
from living_harness.memory.vector_store import VectorStore
from living_harness.memory.consolidation import MemoryConsolidator

def test_reflection_consolidation():
    db_path = "test_reflection_memory.json"
    if os.path.exists(db_path):
        os.remove(db_path)
    vs = VectorStore(db_path=db_path)

    vs.add_item("Thought 1", [0.1, 0.2], {"type": "internal_monologue"})
    vs.add_item("Thought 2", [0.11, 0.19], {"type": "internal_monologue"})

    consolidator = MemoryConsolidator(vector_store=vs)
    consolidator.run_sleep_cycle()

    assert len(vs.memory) > 0
    assert any(m["metadata"]["type"] == "meta_knowledge" for m in vs.memory)
    assert not any(m["metadata"]["type"] == "internal_monologue" for m in vs.memory)

    if os.path.exists(db_path):
        os.remove(db_path)
