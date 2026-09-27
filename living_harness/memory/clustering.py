import math
import random
from typing import List, Tuple

def euclidean_distance(v1: List[float], v2: List[float]) -> float:
    """Вычисляет евклидово расстояние между двумя векторами."""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))

def cosine_distance(v1: List[float], v2: List[float]) -> float:
    """Вычисляет косинусное расстояние между двумя векторами."""
    dot_product = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a ** 2 for a in v1))
    norm2 = math.sqrt(sum(b ** 2 for b in v2))
    if norm1 == 0 or norm2 == 0:
        return 1.0
    return 1.0 - (dot_product / (norm1 * norm2))

def k_means_clustering(vectors: List[List[float]], k: int = 3, max_iters: int = 100, use_cosine: bool = True) -> Tuple[List[List[float]], List[int]]:
    """Выполняет кластеризацию векторов алгоритмом K-Means с инициализацией KMeans++."""
    if not vectors:
        return [], []

    n = len(vectors)
    if k > n:
        k = n

    random.seed(42)
    distance_fn = cosine_distance if use_cosine else euclidean_distance

    # KMeans++ initialization
    centroids = [random.choice(vectors)]
    for _ in range(1, k):
        distances = []
        for v in vectors:
            min_dist = min(distance_fn(v, c) for c in centroids)
            distances.append(min_dist ** 2)

        total_dist = sum(distances)
        if total_dist == 0:
            centroids.append(random.choice(vectors))
            continue

        probs = [d / total_dist for d in distances]
        r = random.random()
        cumulative = 0.0
        for i, p in enumerate(probs):
            cumulative += p
            if r <= cumulative:
                centroids.append(vectors[i])
                break

    if len(centroids) < k:
        centroids.extend(random.choices(vectors, k=k-len(centroids)))

    labels = [0] * n

    for _ in range(max_iters):
        new_labels = []
        for v in vectors:
            distances_to_centroids = [distance_fn(v, c) for c in centroids]
            new_labels.append(distances_to_centroids.index(min(distances_to_centroids)))

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

        if use_cosine:
            for i in range(k):
                norm = math.sqrt(sum(a ** 2 for a in new_centroids[i]))
                if norm > 0:
                    new_centroids[i] = [val / norm for val in new_centroids[i]]

        if labels == new_labels:
            break
        labels = new_labels
        centroids = new_centroids

    return centroids, labels
