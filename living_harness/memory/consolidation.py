import time
from typing import List, Dict, Any
from living_harness.memory.clustering import k_means_clustering

class MemoryConsolidator:
    def __init__(self, vector_store=None, num_clusters: int = 3):
        self.vector_store = vector_store
        self.num_clusters = num_clusters

    def consolidate_memories(self, memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Анализирует эпизодические рассуждения и агрегирует их в абстрактные правила.
        В реальной системе здесь будет LLM-вызов для суммаризации и обобщения.
        """
        if not memories:
            return []

        vectors = []
        valid_memories = []
        for m in memories:
            if "vector" in m and isinstance(m["vector"], list):
                vectors.append(m["vector"])
                valid_memories.append(m)

        if not vectors:
            aggregated_content = " | ".join([m.get("content", "") for m in memories])
            abstract_rule = {
                "type": "semantic",
                "content": f"Consolidated rule based on: {aggregated_content}",
                "timestamp": time.time(),
                "importance": 0.8
            }
            return [abstract_rule]

        k = min(self.num_clusters, len(vectors))
        centroids, labels = k_means_clustering(vectors, k=k)

        clustered_memories = {i: [] for i in range(len(centroids))}
        for idx, label in enumerate(labels):
            clustered_memories[label].append(valid_memories[idx])

        abstract_rules = []
        for i, cluster in clustered_memories.items():
            if not cluster:
                continue
            aggregated_content = " | ".join([m.get("content", "") for m in cluster])
            abstract_rule = {
                "type": "semantic",
                "content": f"Consolidated rule for cluster {i} based on: {aggregated_content}",
                "centroid": centroids[i],
                "timestamp": time.time(),
                "importance": 0.8
            }
            abstract_rules.append(abstract_rule)

        return abstract_rules

    def run_sleep_cycle(self):
        """
        Фоновый процесс "сна" для перевода эпизодической памяти в семантическую.
        """
        pass
