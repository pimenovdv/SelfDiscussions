import json
import os
import math
import tempfile
from living_harness.memory.vector_store import VectorStore

def test_dissonance_resolution():
    fd, temp_path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        vs = VectorStore(db_path=temp_path)

        # 1. Add conflicting concepts
        vs.add_item("Order is strictly necessary for survival.", [0.9, 0.9, 0.9], domain="evolution")
        vs.add_item("Chaos is strictly necessary for survival.", [-0.9, -0.9, -0.9], domain="evolution")

        # Call synthesize_resolution
        new_abstractions = vs.synthesize_resolution(domain="evolution", threshold=-0.5)

        assert len(new_abstractions) > 0, "Failed to synthesize resolution"

        # Calculate metrics
        metrics = {
            "abstractions_generated": len(new_abstractions),
            "source_pairs": len(new_abstractions),
            "threshold_used": -0.5,
            "domain": "evolution",
            "meta_levels": [item["metadata"]["meta_level"] for item in new_abstractions]
        }

        # Determine the project root to reliably save data
        # __file__ is living_harness/experiments/test_dissonance_resolution.py
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        data_dir = os.path.join(project_root, "living_harness", "data")

        # Ensure data directory exists
        os.makedirs(data_dir, exist_ok=True)

        # Save metrics
        metrics_file = os.path.join(data_dir, "resolution_metrics.json")
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=4)

        print(f"Synthesized {len(new_abstractions)} abstractions. Metrics saved to {metrics_file}")

    finally:
        os.remove(temp_path)

if __name__ == "__main__":
    test_dissonance_resolution()
