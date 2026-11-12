import sys
import os
import json
import tempfile
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from living_harness.memory.vector_store import VectorStore

def run_test():
    _, db_path = tempfile.mkstemp(suffix=".json")
    try:
        store = VectorStore(db_path=db_path)

        # 1. Create a dense cluster (thesis)
        # The thesis is [1.0, 0.0, 0.0]
        for i in range(10):
            # Add very small noise to create a cluster
            vec = [1.0, i*0.001, 0.0]
            store.add_item(text=f"Thesis variant {i}", vector=vec, domain="test")

        # 2. Create a single conflicting item (anti-thesis)
        # The anti-thesis is [-1.0, 0.0, 0.0]
        store.add_item(text="Anti-thesis", vector=[-1.0, 0.0, 0.0], domain="test")

        # 3. Trigger resolution
        abstractions = store.synthesize_resolution(domain="test", threshold=-0.5)

        # Analyze results
        print(f"Generated {len(abstractions)} abstractions.")

        if abstractions:
            # Get the abstraction for Thesis variant 0 and Anti-thesis
            abs_vec = abstractions[0]["vector"]
            print(f"First abstraction vector: {abs_vec}")

            metrics = {
                "num_abstractions": len(abstractions),
                "abstraction_vector": abs_vec,
                "shifted_towards_dense_cluster": abs_vec[0] > 0
            }

            os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data"), exist_ok=True)
            with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/nonlinear_resolution_metrics.json"), "w") as f:
                json.dump(metrics, f)

            print("Metrics saved.")
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)

if __name__ == "__main__":
    run_test()
