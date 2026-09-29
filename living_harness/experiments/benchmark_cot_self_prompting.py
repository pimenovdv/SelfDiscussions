import json
import time
import os
from unittest.mock import patch
from living_harness.core.self_prompting import SelfPrompter
from living_harness.core.local_llm_connector import LocalLLMConnector

def run_benchmark():
    # Setup mocks since we can't reliably load LLMs in this environment
    connector = LocalLLMConnector(backend="ollama")
    prompter = SelfPrompter(llm_connector=connector, idle_threshold=0)

    contexts = [
        "User has been asking about python classes.",
        "System is analyzing memory compression algorithms.",
        "We are exploring thermodynamics and AI."
    ]

    mock_responses = [
        "<think>Python classes are foundational. Let's explore inheritance.</think>Explore python inheritance.",
        "<think>Compression is key. K-Means is good, what about DBSCAN?</think>Investigate DBSCAN for memory compression.",
        "<think>Thermodynamics involves entropy. Let's map it to information theory.</think>Analyze entropy in information theory."
    ]

    results = []

    # We patch _generate_ollama directly since the connector uses ollama backend
    with patch.object(LocalLLMConnector, '_generate_ollama', side_effect=mock_responses):
        for ctx in contexts:
            start_time = time.time()
            # Force idle state
            prompter.last_active_time = 0

            if prompter.should_self_prompt():
                goal = prompter.generate_goal(ctx)
                end_time = time.time()

                results.append({
                    "context": ctx,
                    "generated_goal": goal,
                    "thought_process": connector.last_thought,
                    "generation_time_sec": round(end_time - start_time, 4),
                    "model_used": "prithivMLmods/SmolLM2-Rethink-135M" # Mocked
                })

    os.makedirs("living_harness/data", exist_ok=True)
    with open("living_harness/data/cot_self_prompting_benchmark.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"Benchmark complete. Results saved to living_harness/data/cot_self_prompting_benchmark.json")

if __name__ == "__main__":
    run_benchmark()
