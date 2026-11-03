import pytest
from living_harness.memory.clustering import adaptive_clustering

def test_adaptive_clustering():
    # 8 vectors, 4 around [1, 0, 0] and 4 around [0, 1, 0]
    vectors = [
        [1.0, 0.1, 0.0],
        [1.0, 0.0, 0.1],
        [0.9, 0.1, 0.0],
        [0.9, 0.0, 0.1],
        [0.0, 1.0, 0.1],
        [0.1, 1.0, 0.0],
        [0.0, 0.9, 0.1],
        [0.1, 0.9, 0.0],
    ]

    # Expected k = sqrt(8/2) = sqrt(4) = 2.
    centroids, labels = adaptive_clustering(vectors, use_cosine=False)

    assert len(centroids) == 2
    assert len(labels) == 8

    label_group1 = labels[:4]
    label_group2 = labels[4:]

    assert len(set(label_group1)) == 1
    assert len(set(label_group2)) == 1
    assert label_group1[0] != label_group2[0]
