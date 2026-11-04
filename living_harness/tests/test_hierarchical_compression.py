import pytest
import random
import math
from memory.clustering import hierarchical_clustering, cosine_distance

def test_hierarchical_compression_accuracy_loss():
    # Создадим набор из 10 близких векторов
    vectors = []
    for _ in range(10):
        vectors.append([1.0 + random.uniform(-0.1, 0.1), 0.0 + random.uniform(-0.1, 0.1)])

    # Запрос
    query = [1.0, 0.0]

    # Расстояние до исходных векторов
    original_distances = [cosine_distance(query, v) for v in vectors]
    avg_original_distance = sum(original_distances) / len(original_distances)

    # Сжатие с max_depth=0 заставит сразу сжать
    tree = hierarchical_clustering(vectors, max_depth=0, use_cosine=True)

    compressed_vector = tree["compressed_vector"]
    compressed_distance = cosine_distance(query, compressed_vector)

    # Потеря точности (разница расстояний) должна быть небольшой
    loss = abs(avg_original_distance - compressed_distance)

    assert loss < 0.1  # Потеря должна быть незначительной
