import random

class EntropyManager:
    """
    Управляет индексом стагнации (Boredom Index) и генерирует возмущения (энтропию)
    для вывода системы из локальных минимумов при длительном отсутствии новой информации.
    """
    def __init__(self, boredom_threshold=5.0):
        self.boredom_index = 0.0
        self.boredom_threshold = boredom_threshold

    def increase_boredom(self, amount=1.0):
        """Увеличивает индекс скуки."""
        self.boredom_index += amount

    def reset_boredom(self):
        """Сбрасывает индекс скуки (например, после получения новой полезной информации)."""
        self.boredom_index = 0.0

    def should_inject_entropy(self) -> bool:
        """Проверяет, достигнут ли порог для генерации возмущения."""
        return self.boredom_index >= self.boredom_threshold

    def generate_perturbation(self) -> str:
        """Генерирует случайное возмущение и сбрасывает индекс."""
        perturbations = [
            "What if we completely invert our current assumptions?",
            "Injecting entropy: switch to a radically different topic.",
            "Consider the problem from a biological rather than mathematical perspective.",
            "Challenge the status quo: find a flaw in the current consensus.",
            "Propose a wild, untested hypothesis to stimulate new ideas."
        ]
        self.reset_boredom()
        return random.choice(perturbations)
