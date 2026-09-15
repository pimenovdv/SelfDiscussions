import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from living_harness.core.agent import Agent
from living_harness.core.local_llm_connector import LocalLLMConnector

def run_experiment():
    print("Начало эксперимента: комплексное тестирование агентов в диалоговой среде")

    # Использование компактной модели из models.md
    connector = LocalLLMConnector(backend="ollama")
    # Переопределяем метод генерации с указанием модели, если нужно
    original_generate = connector.generate
    connector.generate = lambda prompt, **kwargs: original_generate(prompt, model="prithivMLmods/SmolLM2-Rethink-360M", **kwargs)

    agent1 = Agent(
        name="Alice (Biologist)",
        system_prompt="You are Alice, a biologist focusing on evolutionary systems.",
        surprise_threshold=0.3,
        llm_connector=connector
    )

    agent2 = Agent(
        name="Bob (AI Researcher)",
        system_prompt="You are Bob, an AI researcher focusing on neural architectures.",
        surprise_threshold=0.3,
        llm_connector=connector
    )

    dialog = []

    current_message = "How can we apply principles of biological evolution to artificial neural networks?"
    dialog.append(f"System: {current_message}")
    print(f"System: {current_message}")

    for i in range(3):
        # Alice processes
        res_alice = agent1.process_input(current_message)
        if res_alice["triggered_burst"]:
            dialog.append(f"[Alice BURST]: {res_alice['burst_output']}")
            print(f"[Alice BURST]: {res_alice['burst_output']}")

        reply_alice = res_alice["response"]
        dialog.append(f"Alice: {reply_alice}")
        print(f"Alice: {reply_alice}")

        current_message = reply_alice

        # Bob processes
        res_bob = agent2.process_input(current_message)
        if res_bob["triggered_burst"]:
            dialog.append(f"[Bob BURST]: {res_bob['burst_output']}")
            print(f"[Bob BURST]: {res_bob['burst_output']}")

        reply_bob = res_bob["response"]
        dialog.append(f"Bob: {reply_bob}")
        print(f"Bob: {reply_bob}")

        current_message = reply_bob

    os.makedirs("living_harness/data", exist_ok=True)
    with open("living_harness/data/agent_dialog_results.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(dialog))

    print("Эксперимент завершен. Результаты сохранены в data/agent_dialog_results.txt")

if __name__ == "__main__":
    run_experiment()
