from living_harness.memory.vector_store import VectorStore

store = VectorStore("living_harness/data/test_vector_memory.json")
store.clear()

store.add_item("The quick brown fox jumps over the lazy dog", [1.0, 0.0, 0.0])
store.add_item("A fast brown fox leaped over a sleepy dog", [0.9, 0.1, 0.0])
store.add_item("Artificial intelligence is fascinating", [0.0, 1.0, 0.0])

results = store.search([1.0, 0.0, 0.0], top_k=2)
print([r["text"] for r in results])
