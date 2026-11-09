import pytest
import time
from living_harness.memory.vector_store import VectorStore

def test_retrieve_by_time():
    store = VectorStore()

    current = time.time()

    # Adding items with explicit timestamps (simulating past and present)
    store.add_item("Past event", [0.1, 0.2], metadata={"timestamp": current - 3600}) # 1 hour ago
    store.add_item("Recent event", [0.2, 0.3], metadata={"timestamp": current - 60}) # 1 minute ago
    store.add_item("Current event", [0.3, 0.4], metadata={"timestamp": current}) # now

    # Verify auto timestamp addition works
    store.add_item("Auto timestamp event", [0.4, 0.5])
    assert "timestamp" in store.memory[-1]["metadata"]
    assert store.memory[-1]["metadata"]["timestamp"] >= current

    # Retrieve last 10 minutes
    results = store.retrieve_by_time(current - 600, current + 10)

    assert len(results) == 3 # Recent, Current, Auto

    # Verify sorting (descending by time)
    assert results[0]["text"] == "Auto timestamp event"
    assert results[1]["text"] == "Current event"
    assert results[2]["text"] == "Recent event"

    # Retrieve past event
    past_results = store.retrieve_by_time(current - 4000, current - 3000)
    assert len(past_results) == 1
    assert past_results[0]["text"] == "Past event"
