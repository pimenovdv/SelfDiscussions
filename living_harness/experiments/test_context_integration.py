from living_harness.memory.context_manager import ContextManager
import os

if os.path.exists("living_harness/data/vector_memory.json"):
    os.remove("living_harness/data/vector_memory.json")

manager = ContextManager("System prompt", max_window_tokens=100)
for i in range(10):
    manager.add_reasoning(f"Some long reasoning block {i} that takes up space.", estimated_tokens=20, vector=[1.0, 0.0, 0.0])

print("Vector store size:", len(manager.vector_store.memory))
retrieved = manager.retrieve_relevant_memory([1.0, 0.0, 0.0])
print("Retrieved:\n" + retrieved)
