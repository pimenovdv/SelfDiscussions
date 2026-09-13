import math
import random
from typing import List

class RelevanceDecay:
    """
    Класс для имитации "морфологического некро-вычисления" (архитектуры увядания).
    Реализует математический штраф и шум за доступ к старым данным,
    имитируя тепловую деградацию.
    """

    def __init__(self, base_decay_rate: float = 0.05, noise_variance_factor: float = 0.01):
        self.base_decay_rate = base_decay_rate
        self.noise_variance_factor = noise_variance_factor

    def calculate_penalty(self, age: float) -> float:
        """
        Рассчитывает штраф за возраст записи (экспоненциальное затухание).
        age: время, прошедшее с момента создания записи.
        Возвращает множитель релевантности (от 0 до 1).
        """
        if age < 0:
            return 1.0
        return math.exp(-self.base_decay_rate * age)

    def apply_thermal_noise(self, vector: List[float], age: float) -> List[float]:
        """
        Применяет "тепловой шум" к векторному представлению (имитация деградации).
        С возрастом дисперсия шума увеличивается.
        """
        if age <= 0:
            return vector

        noise_std = math.sqrt(age) * self.noise_variance_factor
        noisy_vector = []
        for val in vector:
            noise = random.gauss(0, noise_std)
            noisy_vector.append(val + noise)

        return noisy_vector
