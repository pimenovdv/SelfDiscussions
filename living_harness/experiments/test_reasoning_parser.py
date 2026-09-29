import pytest
from living_harness.core.reasoning_parser import ReasoningParser

def test_reasoning_parser_with_tags():
    text = "<think>\nThis is a thought process.\nStep 1.\n</think>\nHere is the final answer."
    thought, answer = ReasoningParser.parse(text)
    assert thought == "This is a thought process.\nStep 1."
    assert answer == "Here is the final answer."

def test_reasoning_parser_no_tags():
    text = "Just a direct answer without thinking."
    thought, answer = ReasoningParser.parse(text)
    assert thought is None
    assert answer == "Just a direct answer without thinking."

def test_reasoning_parser_unclosed_tag():
    text = "<think>Wait, I didn't finish"
    thought, answer = ReasoningParser.parse(text)
    assert thought == "Wait, I didn't finish"
    assert answer == ""
