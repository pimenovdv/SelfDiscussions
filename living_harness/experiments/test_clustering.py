import pytest
from living_harness.memory.clustering import k_means_clustering

def test_k_means_clustering():
    vectors = [
        [1.0, 0.0, 0.0],
        [0.9, 0.1, 0.0],
        [0.0, 1.0, 0.0],
        [0.1, 0.9, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.1, 0.9]
    ]
    centroids, labels = k_means_clustering(vectors, k=3)
    assert len(centroids) == 3
    assert len(labels) == 6

    # Check if elements that are close to each other are in the same cluster
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[4] == labels[5]

    # Check if clusters are distinct
    assert labels[0] != labels[2]
    assert labels[0] != labels[4]
    assert labels[2] != labels[4]

def test_k_means_clustering_empty():
    centroids, labels = k_means_clustering([], k=3)
    assert centroids == []
    assert labels == []

def test_k_means_clustering_fewer_vectors_than_k():
    vectors = [[1.0, 0.0], [0.0, 1.0]]
    centroids, labels = k_means_clustering(vectors, k=3)
    assert len(centroids) == 2
    assert len(labels) == 2
    assert labels[0] != labels[1]
