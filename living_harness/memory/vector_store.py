import json
import os
import math
from typing import List, Dict, Any, Optional

class VectorStore:
    def __init__(self, db_path: str = "living_harness/data/vector_memory.json"):
        self.db_path = db_path
        self.memory: List[Dict[str, Any]] = []
        self._load()

    def _load(self):
        if os.path.exists(self.db_path):
            with open(self.db_path, "r", encoding="utf-8") as f:
                try:
                    self.memory = json.load(f)
                except json.JSONDecodeError:
                    self.memory = []
        else:
            self.memory = []

    def _save(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with open(self.db_path, "w", encoding="utf-8") as f:
            json.dump(self.memory, f, ensure_ascii=False, indent=2)

    def add_item(self, text: str, vector: List[float], metadata: Optional[Dict[str, Any]] = None):
        """Добавляет элемент в векторное хранилище."""
        if metadata is None:
            metadata = {}
        item = {
            "text": text,
            "vector": vector,
            "metadata": metadata
        }
        self.memory.append(item)
        self._save()

    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        if not v1 or not v2 or len(v1) != len(v2):
            return 0.0
        dot_product = sum(a * b for a, b in zip(v1, v2))
        norm_v1 = math.sqrt(sum(a * a for a in v1))
        norm_v2 = math.sqrt(sum(b * b for b in v2))
        if norm_v1 == 0.0 or norm_v2 == 0.0:
            return 0.0
        return dot_product / (norm_v1 * norm_v2)

    def search(self, query_vector: List[float], top_k: int = 3) -> List[Dict[str, Any]]:
        """Ищет top_k наиболее похожих элементов по косинусному сходству."""
        scored_items = []
        for item in self.memory:
            if "vector" in item and item["vector"]:
                score = self._cosine_similarity(query_vector, item["vector"])
                scored_items.append({"score": score, "item": item})

        scored_items.sort(key=lambda x: x["score"], reverse=True)
        return [scored["item"] for scored in scored_items[:top_k]]

    def clear(self):
        self.memory = []
        self._save()
