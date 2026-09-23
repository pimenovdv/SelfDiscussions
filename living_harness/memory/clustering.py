import math
import random
from typing import List, Tuple

def euclidean_distance(v1: List[float], v2: List[float]) -> float:
    """Вычисляет евклидово расстояние между двумя векторами."""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))

def k_means_clustering(vectors: List[List[float]], k: int = 3, max_iters: int = 100) -> Tuple[List[List[float]], List[int]]:
    """Выполняет кластеризацию векторов алгоритмом K-Means."""
    if not vectors:
        return [], []

    n = len(vectors)
    if k > n:
        k = n

    random.seed(42)
    centroids = random.sample(vectors, k)

    labels = [0] * n

    for _ in range(max_iters):
        new_labels = []
        for v in vectors:
            distances = [euclidean_distance(v, c) for c in centroids]
            new_labels.append(distances.index(min(distances)))

        new_centroids = [[0.0] * len(vectors[0]) for _ in range(k)]
        counts = [0] * k

        for i, label in enumerate(new_labels):
            for j in range(len(vectors[i])):
                new_centroids[label][j] += vectors[i][j]
            counts[label] += 1

        for i in range(k):
            if counts[i] > 0:
                new_centroids[i] = [val / counts[i] for val in new_centroids[i]]
            else:
                new_centroids[i] = random.choice(vectors)

        if labels == new_labels:
            break
        labels = new_labels
        centroids = new_centroids

    return centroids, labels
