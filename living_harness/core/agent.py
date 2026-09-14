import time
from typing import Optional, Dict, Any

from living_harness.core.burst_inference import BurstInferenceManager
from living_harness.memory.context_manager import ContextManager

class Agent:
    def __init__(self, name: str, system_prompt: str, surprise_threshold: float = 0.7):
        self.name = name
        self.context_manager = ContextManager(system_prompt=system_prompt)
        self.burst_manager = BurstInferenceManager(surprise_threshold=surprise_threshold)

    def process_input(self, user_input: str) -> Dict[str, Any]:
        """
        Обрабатывает новый ввод. Запускает 'burst of thought' (прерывистое вычисление),
        если уровень новизны (сюрприза) превышает порог.
        """
        current_context = self.context_manager.build_prompt()
        result = {"triggered_burst": False, "burst_output": None, "response": None}

        # Проверяем, нужен ли всплеск размышлений (burst)
        if self.burst_manager.should_trigger_burst(current_context, user_input):
            result["triggered_burst"] = True
            burst_output = self.burst_manager.trigger_burst(user_input)
            result["burst_output"] = burst_output
            # Оцениваем количество токенов приблизительно как длину строки / 4
            self.context_manager.add_reasoning(burst_output, len(burst_output) // 4)

        # Генерация финального ответа (заглушка для реального вызова LLM)
        response = f"[{self.name}] Ответ на: {user_input[:20]}..."
        result["response"] = response

        # Добавляем в память
        self.context_manager.add_memory(f"User: {user_input}")
        self.context_manager.add_memory(f"Agent: {response}")

        return result
