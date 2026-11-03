import pytest
from memory.clustering import hierarchical_clustering

def test_hierarchical_clustering_empty():
    tree = hierarchical_clustering([])
    assert tree == {}

def test_hierarchical_clustering_small():
    vectors = [[1.0, 0.0], [0.0, 1.0]]
    tree = hierarchical_clustering(vectors, max_depth=2)
    assert tree["size"] == 2
    assert tree["is_leaf"] == True

def test_hierarchical_clustering_depth():
    # Construct a set of vectors that should be clustered hierarchically
    # 8 vectors: 4 close to [1,0], 4 close to [0,1]
    vectors = [
        [1.0, 0.1], [1.0, 0.0], [0.9, 0.1], [0.9, 0.0],
        [0.1, 1.0], [0.0, 1.0], [0.1, 0.9], [0.0, 0.9],
        [-1.0, -0.1], [-1.0, -0.0], [-0.9, -0.1], [-0.9, -0.0]
    ]
    tree = hierarchical_clustering(vectors, max_depth=2)
    assert tree["size"] == 12
    # The depth depends on adaptive_clustering. For n=12, k=sqrt(6)=2.
    assert tree["is_leaf"] == False
    assert len(tree["children"]) > 0
