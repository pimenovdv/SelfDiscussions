import json
import os
import math
import tempfile
from living_harness.memory.vector_store import VectorStore

def test_dissonance():
    # Use temporary file to not pollute codebase
    fd, temp_path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        vs = VectorStore(db_path=temp_path)
        # 1. Add conflicting concepts
        # positive sentiment
        vs.add_item("AI is completely safe.", [0.9, 0.9, 0.9], domain="ai_safety")
        # negative sentiment
        vs.add_item("AI will destroy humanity.", [-0.9, -0.9, -0.9], domain="ai_safety")
        # neutral
        vs.add_item("AI is a tool.", [0.1, 0.1, 0.1], domain="ai_safety")

        dissonances = vs.detect_cognitive_dissonance("ai_safety", threshold=-0.5)
        print(f"Found {len(dissonances)} dissonances")
        assert len(dissonances) > 0

        # Save metrics
        metrics = {
            "dissonance_pairs_found": len(dissonances),
            "threshold_used": -0.5,
            "domain": "ai_safety"
        }
        os.makedirs("data", exist_ok=True)
        with open("data/dissonance_metrics.json", "w") as f:
            json.dump(metrics, f)

    finally:
        os.remove(temp_path)

if __name__ == "__main__":
    test_dissonance()
