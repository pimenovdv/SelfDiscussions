import pytest
import datetime
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from living_harness.memory.context_manager import ContextManager

def test_memory_decay():
    cm = ContextManager(system_prompt="Test")

    # Add a fresh memory
    cm.add_memory("Fresh memory")

    # Add an old memory manually
    old_time = (datetime.datetime.now() - datetime.timedelta(hours=2)).isoformat()
    cm.memory.append({"timestamp": old_time, "content": "Old memory"})

    assert len(cm.memory) == 2

    # build_prompt should trigger decay and remove the old memory
    prompt = cm.build_prompt()

    assert len(cm.memory) == 1
    assert cm.memory[0]["content"] == "Fresh memory"
    assert "Fresh memory" in prompt
    assert "Old memory" not in prompt

def test_decay_threshold():
    # Set high decay rate so it decays quickly
    cm = ContextManager(system_prompt="Test", decay_rate=0.5)

    # 5 minutes old (e^-2.5 = 0.08 < 0.1 threshold)
    old_time = (datetime.datetime.now() - datetime.timedelta(minutes=5)).isoformat()
    cm.memory.append({"timestamp": old_time, "content": "Borderline memory"})

    cm.build_prompt()
    assert len(cm.memory) == 0
