import sys
import os
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from living_harness.core.background_loop import BackgroundLoop
from living_harness.memory.vector_store import VectorStore
from living_harness.memory.consolidation import MemoryConsolidator

def test_full_sleep_cycle():
    # Setup
    db_path = "living_harness/data/test_vector_memory.json"
    if os.path.exists(db_path):
        os.remove(db_path)

    vs = VectorStore(db_path=db_path)
    # Add some episodic memories
    vs.add_item("I learned how to add numbers.", [0.1, 0.2, 0.3], {"type": "episodic"})
    vs.add_item("Addition is commutative.", [0.11, 0.21, 0.31], {"type": "episodic"})
    vs.add_item("I saw a dog.", [0.9, 0.8, 0.7], {"type": "episodic"})
    vs.add_item("Dogs bark.", [0.91, 0.81, 0.71], {"type": "episodic"})

    consolidator = MemoryConsolidator(vector_store=vs, num_clusters=2)

    print("Memories before sleep cycle:")
    for m in vs.memory:
        print(f" - {m['text']} (type: {m['metadata'].get('type')})")

    # Run sleep cycle
    print("\nRunning sleep cycle...")
    consolidator.run_sleep_cycle()

    print("\nMemories after sleep cycle:")
    for m in vs.memory:
        print(f" - {m['text']} (type: {m['metadata'].get('type')})")

    assert len(vs.memory) > 0, "No memories found after sleep cycle."
    assert all(m['metadata'].get('type') == 'semantic' for m in vs.memory), "Not all memories are semantic."
    print("\nSuccess: Sleep cycle successfully consolidated memories!")

if __name__ == "__main__":
    test_full_sleep_cycle()
