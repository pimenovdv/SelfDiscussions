import numpy as np

class NoiseInjector:
    def __init__(self, error_threshold: float = 0.5, base_temperature: float = 0.7, max_temperature: float = 1.5):
        """
        Инициализация модуля инъекции шума на основе ошибки предсказания.

        :param error_threshold: Порог ошибки предсказания (например, доля повторяющихся n-грамм),
                                при котором активируется инъекция шума.
        :param base_temperature: Базовая температура генерации (нормальное состояние).
        :param max_temperature: Максимальная температура при пиковой ошибке (режим "исследования").
        """
        self.error_threshold = error_threshold
        self.base_temperature = base_temperature
        self.max_temperature = max_temperature
        self.current_error = 0.0

    def update_prediction_error(self, recent_tokens: list, window_size: int = 50) -> float:
        """
        Обновляет и возвращает метрику ошибки предсказания.
        В качестве эвристики используем степень зацикливания (уменьшение энтропии/разнообразия),
        так как это свидетельствует о том, что модель не может найти новый путь решения.
        """
        if len(recent_tokens) < window_size:
            self.current_error = 0.0
            return self.current_error

        # Простая эвристика: доля уникальных токенов в окне
        window = recent_tokens[-window_size:]
        unique_tokens = len(set(window))
        diversity_ratio = unique_tokens / window_size

        # Если разнообразие падает (зацикливание), "ошибка предсказания" (отсутствие прогресса) растет
        self.current_error = 1.0 - diversity_ratio
        return self.current_error

    def get_temperature(self) -> float:
        """
        Возвращает текущую температуру генерации на основе накопленной ошибки.
        Если ошибка превышает порог, происходит "впрыскивание" шума через повышение температуры.
        """
        if self.current_error > self.error_threshold:
            # Линейное скалирование температуры от base до max в зависимости от превышения порога
            excess = (self.current_error - self.error_threshold) / (1.0 - self.error_threshold)
            return self.base_temperature + excess * (self.max_temperature - self.base_temperature)
        return self.base_temperature

    def get_logits_noise(self, vocab_size: int, scale: float = 0.1) -> np.ndarray:
        """
        Альтернативный метод: прямое добавление случайного шума к логитам.
        Возвращает вектор шума, который можно прибавить к логитам перед softmax.
        """
        if self.current_error > self.error_threshold:
            excess = (self.current_error - self.error_threshold) / (1.0 - self.error_threshold)
            return np.random.normal(0, scale * excess, vocab_size)
        return np.zeros(vocab_size)
