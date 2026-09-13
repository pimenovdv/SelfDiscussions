import time
from typing import Dict, Any, List

class BurstInferenceManager:
    """
    Управляет прерывистым вычислением (Bursts of thought).
    Инициирует инференс только при превышении порога 'ошибки предсказания' или 'сюрприза'.
    """
    def __init__(self, surprise_threshold: float = 0.7):
        self.surprise_threshold = surprise_threshold
        self.last_burst_time = time.time()
        self.burst_history: List[Dict[str, Any]] = []

    def calculate_surprise(self, current_context: str, new_input: str) -> float:
        """
        Упрощенная эвристика вычисления 'сюрприза' или новизны.
        В реальности здесь должно быть вычисление ошибки предсказания модели.
        """
        # Заглушка: базовое сравнение длины и уникальных слов
        words_ctx = set(current_context.lower().split())
        words_new = set(new_input.lower().split())

        if not words_ctx:
            return 1.0 # Максимальный сюрприз при пустом контексте

        novel_words = words_new - words_ctx
        novelty_ratio = len(novel_words) / max(1, len(words_new))

        return novelty_ratio

    def should_trigger_burst(self, current_context: str, new_input: str) -> bool:
        """
        Определяет, нужно ли запускать блок размышлений (burst).
        """
        surprise = self.calculate_surprise(current_context, new_input)
        return surprise >= self.surprise_threshold

    def trigger_burst(self, prompt: str) -> str:
        """
        Симуляция запуска размышлений. Возвращает токен <think> и логику.
        """
        self.last_burst_time = time.time()
        burst_output = f"<think> Зафиксирован высокий уровень новизны. Анализирую входные данные: '{prompt[:30]}...' </think>"
        self.burst_history.append({"time": self.last_burst_time, "prompt": prompt})
        return burst_output
