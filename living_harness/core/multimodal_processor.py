import json

class MultimodalProcessor:
    def __init__(self):
        self.modality_weights = {
            "text": 1.0,
            "image": 1.5,
            "audio": 1.2
        }

    def process_signal(self, modality: str, raw_data: str) -> dict:
        """
        Процессинг нетекстового сигнала и расчет базового веса 'сюрприза'.
        В реальной системе здесь будет вызов соответствующих энкодеров.
        """
        weight = self.modality_weights.get(modality, 1.0)

        # Симуляция извлечения фичей и расчета энтропии
        simulated_entropy = len(raw_data) * 0.01 * weight

        return {
            "modality": modality,
            "simulated_entropy": simulated_entropy,
            "processed_features": f"encoded_{modality}_features"
        }

    def calculate_multimodal_surprise(self, signals: list) -> float:
        """
        Агрегация 'сюрприза' из нескольких модальностей.
        """
        total_surprise = 0.0
        for signal in signals:
            processed = self.process_signal(signal['modality'], signal['data'])
            total_surprise += processed['simulated_entropy']

        return min(total_surprise, 1.0) # Нормализация
