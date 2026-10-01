import pytest
from living_harness.core.self_prompting import SelfPrompter
from living_harness.core.attention_collapse_penalty import detect_and_penalize_repetition

class MockLLMConnector:
    def __init__(self, responses):
        self.responses = responses
        self.call_count = 0
        self.last_thought = None

    def generate(self, prompt, **kwargs):
        if self.call_count < len(self.responses):
            resp = self.responses[self.call_count]
            self.call_count += 1
            return resp
        return "I am stuck in a loop."

def test_detect_and_penalize_repetition():
    assert detect_and_penalize_repetition("I think therefore I am", "I think therefore I") == True
    assert detect_and_penalize_repetition("A completely different thought", "I think therefore I am") == False
    assert detect_and_penalize_repetition("Hello world", "") == False

def test_self_prompter_attention_collapse_protection():
    mock_llm = MockLLMConnector([
        "Explore abstract concepts of time.",
        "Explore abstract concepts of time and space.",
        "Explore abstract concepts of time and space and energy."
    ])

    prompter = SelfPrompter(llm_connector=mock_llm)

    # First call - should be normal
    goal1 = prompter.generate_goal("Context summary")
    assert goal1 == "Explore abstract concepts of time."

    # Second call - highly repetitive, should trigger penalty
    goal2 = prompter.generate_goal("Context summary")
    assert goal2 == "Explore alternative abstract concepts due to repetition detected."

    # Third call - model tries to repeat again, should trigger penalty
    goal3 = prompter.generate_goal("Context summary")
    assert goal3 == "Explore alternative abstract concepts due to repetition detected."
