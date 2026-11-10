import json
import os
import sys
import tempfile
import random

# Ensure PYTHONPATH is set correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from living_harness.memory.vector_store import VectorStore

def test_uncertain_goal_progression():
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)

    vs = VectorStore(db_path=path)

    goal_vector = [1.0, 0.0, 0.0, 0.0, 0.0]
    vs.set_active_goal("test_uncertain_goal", goal_vector)

    vec = vs.get_uncertain_goal_vector(noise_level=0.1)
    assert len(vec) == len(goal_vector), "Dimensions do not match"
    assert vec != goal_vector, "Noise was not added correctly"

    metrics = {
        "status": "success",
        "description": "Simulation for uncertain goal progression",
        "goal_variance": 0.1,
        "original_vector": goal_vector,
        "noisy_vector": vec
    }

    os.makedirs(os.path.join(os.path.dirname(__file__), "../data"), exist_ok=True)
    with open(os.path.join(os.path.dirname(__file__), "../data/uncertain_goal_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    os.remove(path)

if __name__ == "__main__":
    test_uncertain_goal_progression()
