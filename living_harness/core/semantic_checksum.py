import re
from typing import Set

class SemanticChecksum:
    """
    Механизм семантического контрольного суммирования для защиты памяти от деградации.
    Оценивает согласованность фактов при перезаписи памяти.
    """

    @staticmethod
    def extract_keywords(text: str) -> Set[str]:
        """Извлекает псевдо-семантические ключевые слова (токены > 4 символов)."""
        words = re.findall(r'\b\w{5,}\b', text.lower())
        return set(words)

    @staticmethod
    def calculate_consistency(old_text: str, new_text: str) -> float:
        """
        Рассчитывает метрику согласованности (0.0 - 1.0) на основе пересечения ключевых слов.
        """
        old_keywords = SemanticChecksum.extract_keywords(old_text)
        new_keywords = SemanticChecksum.extract_keywords(new_text)

        if not old_keywords:
            return 1.0 # Если старый текст был пуст, согласованность 100%

        intersection = old_keywords.intersection(new_keywords)
        consistency = len(intersection) / len(old_keywords)
        return consistency

    @staticmethod
    def verify_overwrite(old_text: str, new_text: str, threshold: float = 0.3) -> bool:
        """
        Проверяет, можно ли перезаписать память.
        Если согласованность ниже порога, считается, что происходит семантический сдвиг (галлюцинация).
        """
        consistency = SemanticChecksum.calculate_consistency(old_text, new_text)
        return consistency >= threshold
