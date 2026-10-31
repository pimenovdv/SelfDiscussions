import pytest
from memory.vector_store import VectorStore

def test_spreading_activation():
    # Инициализация с in-memory хранилищем для тестов
    vs = VectorStore(":memory:")
    vs.memory = []

    # Создаем тестовые данные
    # Вектор 1 (очень активный)
    vs.memory.append({
        "text": "Concept A (Active)",
        "vector": [1.0, 0.0, 0.0],
        "metadata": {"importance_weight": 3.0}
    })

    # Вектор 2 (близкий сосед)
    vs.memory.append({
        "text": "Concept B (Neighbor)",
        "vector": [0.9, 0.1, 0.0],
        "metadata": {"importance_weight": 1.0}
    })

    # Вектор 3 (далекий сосед)
    vs.memory.append({
        "text": "Concept C (Distant)",
        "vector": [0.0, 1.0, 0.0],
        "metadata": {"importance_weight": 1.0}
    })

    # Запускаем spreading activation
    # spread_factor=0.5, threshold=2.0
    vs.spreading_activation(spread_factor=0.5, threshold=2.0)

    # Проверяем веса
    # Concept A не меняется
    assert vs.memory[0]["metadata"]["importance_weight"] == 3.0

    # Concept B должен получить буст: 1.0 + (3.0 * 0.5 * sim(A, B))
    # sim(A, B) = (1.0*0.9) / (1 * sqrt(0.81+0.01)) = 0.9 / 0.9055 = 0.9938
    # 1.0 + 3.0 * 0.5 * 0.9938 = 1.0 + 1.4908 = 2.4908
    assert vs.memory[1]["metadata"]["importance_weight"] > 1.0

    # Concept C не должен получить буст (sim(A, C) = 0)
    assert vs.memory[2]["metadata"]["importance_weight"] == 1.0
