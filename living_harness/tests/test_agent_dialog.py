import pytest
from living_harness.core.agent import Agent

def test_agent_dialog_environment():
    agent1 = Agent(name="Alice", system_prompt="You are Alice, a helpful assistant.", surprise_threshold=0.5)
    agent2 = Agent(name="Bob", system_prompt="You are Bob, a curious researcher.", surprise_threshold=0.5)

    msg1 = "Hello Bob, what do you think about consciousness?"
    res1 = agent2.process_input(msg1)

    assert res1["triggered_burst"] is True
    assert res1["burst_output"] is not None
    assert res1["response"] is not None
    assert "User: Hello Bob, what do you think about consciousness?" in agent2.context_manager.build_prompt()

    msg2 = "I think it's a byproduct of complex information processing."
    res2 = agent1.process_input(msg2)

    assert res2["triggered_burst"] is True
    assert res2["burst_output"] is not None
    assert res2["response"] is not None
    assert "User: I think it's a byproduct of complex information processing." in agent1.context_manager.build_prompt()
