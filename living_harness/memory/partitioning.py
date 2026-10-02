import time
from typing import List, Dict, Any

class PartitionManager:
    """
    Управляет партиционированием векторного хранилища.
    Разделяет воспоминания на 'hot' (актуальные для консолидации) и 'cold' (архивные).
    """
    def __init__(self, hot_window_seconds: float = 86400.0, max_batch_size: int = 500):
        self.hot_window_seconds = hot_window_seconds
        self.max_batch_size = max_batch_size

    def get_hot_partition(self, memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Извлекает записи, попавшие во временное окно hot_window_seconds,
        или те, у которых отсутствует временная метка (считаются новыми).
        """
        current_time = time.time()
        hot_memories = []
        for m in memories:
            timestamp = m.get("metadata", {}).get("timestamp", current_time)
            if (current_time - timestamp) <= self.hot_window_seconds:
                hot_memories.append(m)
        return hot_memories

    def create_batches(self, memories: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """
        Разбивает список воспоминаний на батчи для пошаговой обработки.
        """
        return [memories[i:i + self.max_batch_size]
                for i in range(0, len(memories), self.max_batch_size)]
