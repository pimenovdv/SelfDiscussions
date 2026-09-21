import re
from typing import List

def calculate_structural_importance(text: str) -> List[float]:
    """
    Вычисляет 'структурную важность' каждого слова/токена.
    Слова в начале предложений, ключевые слова и границы абзацев
    получают больший вес для защиты от вытеснения из KV-кэша.
    """
    # Разбиваем текст на слова и знаки препинания
    words = re.findall(r'\b\w+\b|\S', text)
    importance_scores = []

    for i, word in enumerate(words):
        score = 0.5  # базовый вес

        # Повышаем важность для начала текста или после окончания предложения
        if i == 0 or (i > 0 and words[i-1] in ['.', '!', '?', '\n']):
            score += 0.5

        # Длинные (потенциально более значимые) слова также важнее
        if len(word) > 7:
            score += 0.2

        importance_scores.append(min(1.0, score))

    return importance_scores

def merge_entropy_and_cohesion(entropies: List[float], cohesions: List[float], alpha: float = 0.5) -> List[float]:
    """
    Комбинирует метрику локального 'сюрприза' (энтропии) и структурной связности.
    """
    length = min(len(entropies), len(cohesions))
    return [alpha * entropies[i] + (1 - alpha) * cohesions[i] for i in range(length)]
