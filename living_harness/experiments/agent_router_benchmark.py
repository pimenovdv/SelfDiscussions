import sys
import os
import time

# Добавляем родительскую директорию в PYTHONPATH для импорта living_harness
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from living_harness.core.agent_router import AgentRouter
from living_harness.core.agent import Agent
from living_harness.core.local_llm_connector import LocalLLMConnector

class MockLLMConnector:
    def __init__(self):
        # Эмуляция задержки LLM
        pass
    def generate(self, prompt, temperature=0.1):
        time.sleep(0.5) # Имитация задержки генерации
        if "quantum" in prompt.lower() or "thermodynamics" in prompt.lower():
            return "Alexey"
        elif "biology" in prompt.lower() or "evolution" in prompt.lower():
            return "Vladimir"
        elif "psychology" in prompt.lower() or "bias" in prompt.lower():
            return "Artem"
        return "Organizer"

def create_agent(name, prompt):
    class MockContextManager:
        def __init__(self, p):
            self.system_prompt = p
    agent = Agent(name, "test_system")
    agent.context_manager = MockContextManager(prompt)
    agent.name = name
    return agent

agents = [
    create_agent("Alexey", "Я Алексей, лучший физик. Знаю все про квантовую механику, термодинамику и энергоэффективность."),
    create_agent("Vladimir", "Я Владимир, нейробиолог. Разбираюсь в синапсах, эволюции мозга и биологических нейросетях."),
    create_agent("Artem", "Я Артём, когнитивный психолог. Изучаю когнитивные искажения, статус кво и ошибки восприятия.")
]

router = AgentRouter(agents, llm_connector=MockLLMConnector(), use_embeddings=True)

queries = [
    ("Расскажи про квантовые эффекты в полупроводниках", "Alexey"),
    ("Как работает память в гиппокампе и эволюция мозга?", "Vladimir"),
    ("Почему люди подвержены ошибке выжившего и когнитивным искажениям?", "Artem"),
    ("Что такое энтропия в контексте термодинамики?", "Alexey"),
    ("Как синапсы передают сигналы в биологических сетях?", "Vladimir"),
    ("Объясни эффект Даннинга-Крюгера", "Artem"), # Heuristic might fail here if prompt doesn't have exact words
]

results = {"Heuristic": {"correct": 0, "time": 0.0}, "LLM": {"correct": 0, "time": 0.0}, "Embedding": {"correct": 0, "time": 0.0}}

# Heuristic Benchmark
start = time.time()
for q, expected in queries:
    routed = router._heuristic_route(q)
    if routed and routed.name == expected:
        results["Heuristic"]["correct"] += 1
results["Heuristic"]["time"] = time.time() - start

# LLM Benchmark
start = time.time()
for q, expected in queries:
    routed = router._llm_route(q)
    if routed and routed.name == expected:
        results["LLM"]["correct"] += 1
results["LLM"]["time"] = time.time() - start

# Embedding Benchmark
start = time.time()
for q, expected in queries:
    routed = router._embedding_route(q)
    if routed and routed.name == expected:
        results["Embedding"]["correct"] += 1
results["Embedding"]["time"] = time.time() - start

total_queries = len(queries)

print("Agent Router Benchmark Results:")
print("=" * 40)
for method, data in results.items():
    accuracy = (data["correct"] / total_queries) * 100
    avg_time = data["time"] / total_queries
    print(f"{method:12} | Accuracy: {accuracy:6.2f}% | Avg Time: {avg_time:.4f}s")

with open(os.path.join(os.path.dirname(__file__), '..', 'data', 'router_benchmark_results.txt'), 'w', encoding='utf-8') as f:
    f.write("Agent Router Benchmark Results:\n")
    f.write("=" * 40 + "\n")
    for method, data in results.items():
        accuracy = (data["correct"] / total_queries) * 100
        avg_time = data["time"] / total_queries
        f.write(f"{method:12} | Accuracy: {accuracy:6.2f}% | Avg Time: {avg_time:.4f}s\n")
