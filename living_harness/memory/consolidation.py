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
        if not self.vector_store:
            return

        # 1. Consolidate episodic memories
        episodic_memories = [m for m in self.vector_store.memory if m.get("metadata", {}).get("type") == "episodic"]
        if episodic_memories:
            memories_to_consolidate = [{"content": m["text"], "vector": m["vector"]} for m in episodic_memories]
            abstract_rules = self.consolidate_memories(memories_to_consolidate)

            # Очищаем старые эпизодические воспоминания
            self.vector_store.memory = [m for m in self.vector_store.memory if m.get("metadata", {}).get("type") != "episodic"]

            # Добавляем новые семантические правила
            for rule in abstract_rules:
                self.vector_store.add_item(
                    text=rule["content"],
                    vector=rule.get("centroid", []),
                    metadata={"type": "semantic", "timestamp": rule["timestamp"], "importance": rule["importance"]}
                )

        # 2. Reflect on internal monologue
        internal_monologues = [m for m in self.vector_store.memory if m.get("metadata", {}).get("type") == "internal_monologue"]
        if internal_monologues:
            monologues_to_consolidate = [{"content": m["text"], "vector": m["vector"]} for m in internal_monologues]
            reflections = self.consolidate_memories(monologues_to_consolidate)

            # Очищаем старые внутренние монологи
            self.vector_store.memory = [m for m in self.vector_store.memory if m.get("metadata", {}).get("type") != "internal_monologue"]

            # Добавляем рефлексии как мета-знания
            for reflection in reflections:
                self.vector_store.add_item(
                    text=f"Meta-Reflection: {reflection['content']}",
                    vector=reflection.get("centroid", []),
                    metadata={"type": "meta_knowledge", "timestamp": reflection["timestamp"], "importance": 0.9}
                )

        self.vector_store._save()
