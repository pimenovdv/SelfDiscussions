import os
import sys
import time
import tracemalloc
from pathlib import Path

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent.parent))

from living_harness.core.agent import Agent
from living_harness.core.local_llm_connector import LocalLLMConnector

def run_stress_test(num_iterations: int = 100):
    print("Начало стресс-теста: профилирование памяти и производительности агента")

    # Настраиваем фиктивный коннектор для быстрого теста, чтобы не зависеть от реального локального сервера
    class MockLLMConnector(LocalLLMConnector):
        def __init__(self, backend="ollama", api_base="http://localhost:11434"):
            self.backend = backend
            self.api_base = api_base

        def generate(self, prompt: str, **kwargs) -> str:
            # Имитация небольшой задержки и возврата ответа
            time.sleep(0.01)
            return f"Mock response to: {prompt[:20]}..."

    connector = MockLLMConnector(backend="ollama")

    agent = Agent(
        name="Test Agent",
        system_prompt="You are a test agent.",
        surprise_threshold=0.1,
        llm_connector=connector
    )

    tracemalloc.start()

    results = []

    start_time = time.time()

    for i in range(num_iterations):
        message = f"Test message number {i} for stress testing."

        iter_start = time.time()
        agent.process_input(message)
        iter_end = time.time()

        current, peak = tracemalloc.get_traced_memory()

        results.append(f"Iteration {i}: Memory: {current / 10**6:.2f}MB / Peak: {peak / 10**6:.2f}MB, Time: {iter_end - iter_start:.4f}s")

        if (i + 1) % 10 == 0:
            print(f"Completed {i+1} iterations...")

    end_time = time.time()

    tracemalloc.stop()

    total_time = end_time - start_time
    summary = f"\nTotal time for {num_iterations} iterations: {total_time:.2f}s"
    results.append(summary)
    print(summary)

    os.makedirs("living_harness/data", exist_ok=True)
    with open("living_harness/data/memory_leak_results.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(results))

    print("Стресс-тест завершен. Результаты сохранены в living_harness/data/memory_leak_results.txt")

if __name__ == "__main__":
    run_stress_test(50)
