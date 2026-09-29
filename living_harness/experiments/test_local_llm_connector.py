import pytest
from unittest.mock import patch, MagicMock
from living_harness.core.local_llm_connector import LocalLLMConnector

@patch.object(LocalLLMConnector, '_generate_ollama')
def test_local_llm_connector_reasoning(mock_generate):
    mock_generate.return_value = "<think>\nThinking about this deeply...\n</think>\nThe answer is 42."
    connector = LocalLLMConnector(backend="ollama")
    result = connector.generate("What is the answer?", model="test-model")

    assert result == "The answer is 42."
    assert connector.last_thought == "Thinking about this deeply..."

@patch.object(LocalLLMConnector, '_generate_ollama')
def test_local_llm_connector_no_reasoning(mock_generate):
    mock_generate.return_value = "The answer is 42."
    connector = LocalLLMConnector(backend="ollama")
    result = connector.generate("What is the answer?", model="test-model")

    assert result == "The answer is 42."
    assert connector.last_thought is None
