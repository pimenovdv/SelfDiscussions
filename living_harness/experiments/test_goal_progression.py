import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import json
import tempfile
import pytest
from living_harness.memory.vector_store import VectorStore

def test_goal_progression():
    # Use a temporary file for the database
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp_file:
        db_path = tmp_file.name

    store = VectorStore(db_path=db_path)

    try:
        # Define a goal
        goal_text = "Find water source"
        goal_vector = [0.8, 0.9, 0.1, 0.2] # Arbitrary goal vector
        store.set_active_goal(goal_text, goal_vector)

        # Event 1: Moving towards water (similar vector)
        event1_vec = [0.75, 0.85, 0.15, 0.25]
        store.add_item("Walked north towards river", event1_vec)
        prog_1 = store.evaluate_progress_towards_goal(event1_vec)

        # Event 2: Moving away from water (dissimilar vector)
        event2_vec = [0.1, 0.2, 0.9, 0.8]
        store.add_item("Walked south into desert", event2_vec)
        prog_2 = store.evaluate_progress_towards_goal(event2_vec)

        assert prog_1 > prog_2, f"Progress moving towards goal ({prog_1}) should be higher than moving away ({prog_2})"

        # Mark completed
        store.mark_goal_completed()
        assert store.get_active_goal() is None, "Goal should be marked completed"

        # Save metrics
        os.makedirs("living_harness/data", exist_ok=True)
        with open("living_harness/data/goal_progression_metrics.json", "w") as f:
            json.dump({"progression_towards_water": prog_1, "progression_away": prog_2}, f)

    finally:
        if os.path.exists(db_path):
            os.remove(db_path)

if __name__ == "__main__":
    test_goal_progression()
    print("Goal progression test passed and metrics saved.")
