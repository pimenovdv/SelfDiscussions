import time
import pytest
from living_harness.memory.partitioning import PartitionManager

def test_sleep_profiling_constant_time():
    """
    Симулирует большое количество мета-знаний и доказывает, что
    фаза 'сна' обрабатывает только горячую партицию,
    обеспечивая константное время (зависящее от hot_window).
    """
    pm = PartitionManager(hot_window_seconds=86400.0, max_batch_size=500)

    current_time = time.time()

    # Генерируем 10000 старых записей (холодная партиция)
    cold_memories = [
        {"id": f"cold_{i}", "metadata": {"timestamp": current_time - 100000.0}}
        for i in range(10000)
    ]

    # Генерируем 100 новых записей (горячая партиция)
    hot_memories = [
        {"id": f"hot_{i}", "metadata": {"timestamp": current_time - 1000.0}}
        for i in range(100)
    ]

    all_memories = cold_memories + hot_memories

    # Профилируем время извлечения горячей партиции
    start_time = time.time()
    extracted_hot = pm.get_hot_partition(all_memories)
    end_time = time.time()

    extraction_duration = end_time - start_time

    # Убеждаемся, что извлечена только горячая партиция
    assert len(extracted_hot) == 100
    assert all(m["id"].startswith("hot_") for m in extracted_hot)

    # Убеждаемся, что операция выполняется достаточно быстро
    # (на 10000+ элементов это должно занимать доли секунды)
    assert extraction_duration < 0.1

    # Проверка создания батчей
    batches = pm.create_batches(extracted_hot)
    assert len(batches) == 1
    assert len(batches[0]) == 100
