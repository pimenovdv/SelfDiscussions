import os
import sys
import json
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from living_harness.memory.vector_store import VectorStore

import tempfile
import atexit

def main():
    fd, temp_db_path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    def cleanup():
        if os.path.exists(temp_db_path):
            os.remove(temp_db_path)
    atexit.register(cleanup)

    store = VectorStore(db_path=temp_db_path)

    start_add = time.time()
    # Add text concepts
    store.add_item("Apple", [0.1, 0.2, 0.3], modality="text")
    store.add_item("Car", [0.4, 0.5, 0.6], modality="text")

    # Add multimodal vectors
    store.add_item("image_apple.jpg", [0.11, 0.22, 0.33], modality="image", linked_concept="Apple")
    store.add_item("audio_apple.wav", [0.09, 0.19, 0.29], modality="audio", linked_concept="Apple")
    store.add_item("image_car.jpg", [0.41, 0.51, 0.61], modality="image", linked_concept="Car")
    add_time = time.time() - start_add

    start_search = time.time()
    results = store.search([0.1, 0.2, 0.3], top_k=5)
    search_time = time.time() - start_search

    metrics = {
        "add_time": add_time,
        "search_time": search_time,
        "results_count": len(results)
    }

    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    with open(f"{data_dir}/multimodal_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("Multimodal benchmark complete. Metrics saved.")

    # Clean up literal ":memory:" file created by VectorStore
    if os.path.exists(":memory:"):
        os.remove(":memory:")
    # Clean up path used in vector store init
    # When initialized with ":memory:", os.path.exists(":memory:") is True,
    # but since it's executed from living_harness dir, the file is living_harness/:memory:
    # Actually we just deleted it. Let's make sure VectorStore uses tempfile instead.

if __name__ == "__main__":
    main()
