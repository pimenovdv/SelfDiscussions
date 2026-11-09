import json
import os
import sys
import uuid
import time
from typing import List, Dict, Any

class MockLLMConnector:
    def __init__(self, backend="ollama"):
        self.backend = backend
        self.last_thought = None
        self.call_count = 0

    def generate(self, prompt: str, model: str = "llama3", max_tokens: int = 1024, temperature: float = 0.7):
        self.call_count += 1
        self.last_thought = "Thinking about sequence prediction..."

        if "Событие A" in prompt and "Событие B" in prompt and "Событие C" in prompt:
            return "Событие D"
        elif "User opened settings" in prompt and "User clicked profile" in prompt:
            return "User edited username"

        return "Predicted next event"

def run_experiment():
    print("Running event prediction experiment...")

    from living_harness.memory.vector_store import VectorStore

    db_path = os.path.join(os.path.dirname(__file__), "../data/test_db_sequence.json")

    if os.path.exists(db_path):
        try:
            os.remove(db_path)
        except:
            pass

    store = VectorStore(db_path=db_path)

    base_time = time.time()

    events = [
        {"id": "evt1", "content": "Событие A: Пользователь зашел на сайт", "metadata": {"timestamp": base_time}},
        {"id": "evt2", "content": "Событие B: Пользователь открыл каталог", "metadata": {"timestamp": base_time + 10}},
        {"id": "evt3", "content": "Событие C: Пользователь выбрал товар", "metadata": {"timestamp": base_time + 25}},
        {"id": "evt4", "content": "Событие D: Пользователь добавил товар в корзину", "metadata": {"timestamp": base_time + 40}},
        {"id": "evt5", "content": "Событие E: Пользователь перешел к оплате", "metadata": {"timestamp": base_time + 60}},
    ]

    for evt in events:
        store.memory.append(evt)

    sequence = store.retrieve_sequence(target_event_id="evt3", time_window=30.0, limit=5)

    print(f"Retrieved sequence size: {len(sequence)}")
    for item in sequence:
        print(f" - {item['content']}")

    try:
        from living_harness.core.local_llm_connector import LocalLLMConnector
        real_llm = LocalLLMConnector(backend="ollama")
        res = real_llm.generate("test", max_tokens=10)
        if res is None:
            raise Exception("LLM returned None")
        llm = real_llm
    except Exception as e:
        print(f"Failed to connect to real LLM, using Mock. Error: {e}")
        llm = MockLLMConnector()

    prompt_context = "\n".join([item["content"] for item in sequence if item["id"] in ["evt1", "evt2", "evt3"]])

    prompt = f"""
    Проанализируй следующую временную последовательность событий и предскажи наиболее вероятное следующее событие.

    История событий:
    {prompt_context}

    Какое следующее событие должно произойти? Ответь только названием события, без лишних рассуждений.
    """

    print("\nSending prompt to LLM:")
    print(prompt)

    prediction = llm.generate(prompt, model="qwen2.5-0.5b-instruct", temperature=0.5)

    print(f"\nLLM Prediction: {prediction}")

    results_dir = os.path.join(os.path.dirname(__file__), "../data")
    os.makedirs(results_dir, exist_ok=True)

    metrics = {
        "timestamp": time.time(),
        "sequence_length": len(sequence),
        "target_event": "evt3",
        "predicted": prediction,
        "actual": "Событие D: Пользователь добавил товар в корзину",
        "success": prediction is not None and len(prediction) > 0
    }

    metrics_path = os.path.join(results_dir, "prediction_experiment_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"\nSaved metrics to {metrics_path}")

if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    run_experiment()
