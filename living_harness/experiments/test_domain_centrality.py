import json
import os
import networkx as nx
import sys

# Добавляем корень проекта в PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from memory.vector_store import VectorStore

def test_export_and_centrality():
    store = VectorStore(db_path="data/test_domain_graph.json")
    store.clear()

    # Симулируем переходы
    store.transitions = {
        "AI": {"Philosophy": 5, "Physics": 2},
        "Philosophy": {"AI": 3, "Biology": 4},
        "Physics": {"AI": 1},
        "Biology": {"Philosophy": 2}
    }

    graph_data = store.export_domain_graph()

    # Загружаем в NetworkX
    G = nx.node_link_graph(graph_data)

    # Вычисляем PageRank
    pagerank = nx.pagerank(G, weight='weight')

    print("PageRank centrality:")
    for node, rank in sorted(pagerank.items(), key=lambda x: x[1], reverse=True):
        print(f"  {node}: {rank:.4f}")

    assert "AI" in pagerank
    assert "Philosophy" in pagerank

    # Сохраняем метрики
    metrics = {
        "pagerank": pagerank,
        "nodes": len(G.nodes),
        "edges": len(G.edges)
    }
    os.makedirs("data", exist_ok=True)
    with open("data/domain_centrality_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    print("Test passed and metrics saved to data/domain_centrality_metrics.json")

if __name__ == "__main__":
    test_export_and_centrality()
