import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from living_harness.core.semantic_checksum import SemanticChecksum
from living_harness.core.tools import Tools
from living_harness.memory.context_manager import ContextManager

def test_semantic_checksum():
    old_text = "The quick brown fox jumps over the lazy dog."
    new_text = "The quick brown fox leaps over the lazy dog."
    bad_text = "Space exploration is the new frontier for humanity."

    assert SemanticChecksum.calculate_consistency(old_text, new_text) > 0.5
    assert SemanticChecksum.calculate_consistency(old_text, bad_text) < 0.3

    assert SemanticChecksum.verify_overwrite(old_text, new_text) == True
    assert SemanticChecksum.verify_overwrite(old_text, bad_text) == False

def test_tools_overwrite():
    cm = ContextManager(system_prompt="Test")
    cm.add_memory("The quick brown fox jumps over the lazy dog.")
    tools = Tools(cm)

    # Good overwrite
    result = tools.overwrite_memory(0, "The quick brown fox leaps over the lazy dog.")
    assert "успешно перезаписана" in result
    assert cm.memory[0]["content"] == "The quick brown fox leaps over the lazy dog."

    # Bad overwrite
    result = tools.overwrite_memory(0, "Space exploration is the new frontier for humanity.")
    assert "Семантический сдвиг слишком велик" in result
    assert cm.memory[0]["content"] == "The quick brown fox leaps over the lazy dog."
