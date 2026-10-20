import json
import os
import math
import time
from typing import List, Dict, Any, Optional, Union, Tuple

class VectorStore:
    def __init__(self, db_path: str = "living_harness/data/vector_memory.json"):
        self.db_path = db_path
        self.memory: List[Dict[str, Any]] = []
        # Cache format: query_tuple -> (timestamp, result_list)
        self._search_cache: Dict[Tuple[Any, ...], Tuple[float, List[Dict[str, Any]]]] = {}
        self._cache_ttl = 60.0 # seconds
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
        dir_name = os.path.dirname(self.db_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        with open(self.db_path, "w", encoding="utf-8") as f:
            json.dump(self.memory, f, ensure_ascii=False, indent=2)

    def add_item(self, text: str, vector: List[float], metadata: Optional[Dict[str, Any]] = None, domain: str = "general"):
        """Добавляет элемент в векторное хранилище."""
        if metadata is None:
            metadata = {}
        if "last_accessed" not in metadata:
            metadata["last_accessed"] = time.time()
        item = {
            "domain": domain,
            "text": text,
            "vector": vector,
            "metadata": metadata
        }
        self.memory.append(item)
        self._search_cache.clear() # Invalidate cache on new item
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

    def search(self, query_vector: List[float], top_k: int = 3, domain: Optional[Union[str, List[str]]] = None) -> List[Dict[str, Any]]:
        """Ищет top_k наиболее похожих элементов по косинусному сходству."""

        # Serialize domain for cache key
        if domain is None:
            domain_key = None
        elif isinstance(domain, str):
            domain_key = domain
        else:
            domain_key = tuple(sorted(domain))

        cache_key = (tuple(query_vector), top_k, domain_key)
        current_time = time.time()

        # Check cache
        if cache_key in self._search_cache:
            cache_time, cached_results = self._search_cache[cache_key]
            if current_time - cache_time < self._cache_ttl:
                # Update reinforcement even if cached
                for item in cached_results:
                    if "metadata" not in item:
                        item["metadata"] = {}
                    current_weight = item["metadata"].get("importance_weight", 1.0)
                    item["metadata"]["importance_weight"] = min(current_weight + 0.1, 5.0)
                    item["metadata"]["last_accessed"] = current_time
                if cached_results:
                    self._save()
                return cached_results

        scored_items = []
        for item in self.memory:
            item_domain = item.get("domain", "general")
            if domain is not None:
                if isinstance(domain, str) and item_domain != domain:
                    continue
                elif isinstance(domain, list) and item_domain not in domain:
                    continue
            if "vector" in item and item["vector"]:
                base_score = self._cosine_similarity(query_vector, item["vector"])

                # Применяем importance_weight, если он есть
                importance_weight = item.get("metadata", {}).get("importance_weight", 1.0)

                # Применяем буст за свежесть (Recency)
                last_accessed = item.get("metadata", {}).get("last_accessed", current_time)
                days_passed = max(0.0, (current_time - last_accessed) / (24.0 * 3600.0))
                # Например, буст до 1.2 для совсем свежих, стремящийся к 1.0 со временем
                recency_boost = 1.0 + 0.2 * math.exp(-days_passed)

                final_score = base_score * math.log1p(importance_weight) * recency_boost

                scored_items.append({"score": final_score, "item": item})

        scored_items.sort(key=lambda x: x["score"], reverse=True)

        top_items = [scored["item"] for scored in scored_items[:top_k]]

        # Подкрепление: увеличиваем importance_weight для извлеченных элементов
        for item in top_items:
            if "metadata" not in item:
                item["metadata"] = {}
            current_weight = item["metadata"].get("importance_weight", 1.0)
            item["metadata"]["importance_weight"] = min(current_weight + 0.1, 5.0)
            item["metadata"]["last_accessed"] = current_time

        if top_items:
            self._save()

        # Update cache
        self._search_cache[cache_key] = (current_time, top_items)

        return top_items


    def decay_weights(self, decay_rate: float = 0.1, days_threshold: float = 1.0):
        """Снижает importance_weight для давно не использовавшихся элементов."""
        current_time = time.time()
        changed = False
        for item in self.memory:
            metadata = item.get("metadata")
            if not metadata:
                continue
            last_accessed = metadata.get("last_accessed", current_time)
            days_passed = max(0.0, (current_time - last_accessed) / (24.0 * 3600.0))
            if days_passed > days_threshold:
                current_weight = metadata.get("importance_weight", 1.0)
                if current_weight > 1.0:
                    decay_factor = math.exp(-decay_rate * (days_passed - days_threshold))
                    new_weight = 1.0 + (current_weight - 1.0) * decay_factor
                    metadata["importance_weight"] = max(1.0, new_weight)
                    changed = True

        if changed:
            self._save()

    def clear(self):
        self.memory = []
        self._search_cache.clear()
        self._save()
