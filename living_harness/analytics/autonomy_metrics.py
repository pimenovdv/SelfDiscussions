import re
from typing import List, Dict

def calculate_reasoning_density(text: str) -> float:
    """
    Рассчитывает плотность рассуждений (отношение символов внутри <think> к общему числу символов).
    """
    think_blocks = re.findall(r'<think>(.*?)</think>', text, re.DOTALL)
    think_length = sum(len(block) for block in think_blocks)
    total_length = len(text)
    return think_length / total_length if total_length > 0 else 0.0

def calculate_idle_time_before_interrupt(logs: List[Dict]) -> float:
    """
    Рассчитывает среднее время бездействия (рассуждений) до прерывания (ask_user).
    """
    # Заглушка для расчета на основе логов
    interrupts = [log for log in logs if log.get("action") == "ask_user"]
    return len(interrupts) * 1.5  # Условное значение для примера
