import pytest
import os
import asyncio
from unittest.mock import MagicMock
from living_harness.memory.vector_store import VectorStore

@pytest.fixture
def temp_memory_file(tmp_path):
    filepath = tmp_path / "test_memory.json"
    yield str(filepath)
    if os.path.exists(filepath):
        os.remove(filepath)

@pytest.mark.asyncio
async def test_background_curiosity_cycle(temp_memory_file):
    store = VectorStore(db_path=temp_memory_file)
    store.integrate_hypothesis_pipeline = MagicMock(return_value=[{"text": "mock hypothesis"}])

    # Run the background cycle as a task
    task = asyncio.create_task(
        store.background_curiosity_cycle(
            interval_seconds=0.1
        )
    )

    # Let it run for a short time
    await asyncio.sleep(0.25)

    # Cancel the infinite loop
    task.cancel()

    try:
        await task
    except asyncio.CancelledError:
        pass

    # Ensure it was called multiple times due to the loop
    assert store.integrate_hypothesis_pipeline.call_count >= 2
