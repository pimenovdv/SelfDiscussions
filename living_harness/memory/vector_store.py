import json
import os
import math
import time
from collections import OrderedDict
from typing import List, Dict, Any, Optional, Union, Tuple

class VectorStore:
    def __init__(self, db_path: str = "living_harness/data/vector_memory.json", max_cache_size: int = 100):
        self.db_path = db_path
        self.max_cache_size = max_cache_size
        self.memory: List[Dict[str, Any]] = []
        # Cache format: query_tuple -> (timestamp, result_list)
        # Using OrderedDict to implement LRU Cache
        self._search_cache: OrderedDict[Tuple[Any, ...], Tuple[float, List[Dict[str, Any]]]] = OrderedDict()
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

    def _quantize_vector(self, vector: List[float], num_bits: int = 8) -> List[int]:
        """Квантует вектор, сохраняя только num_bits бит (в данном случае просто в int от 0 до 255).
        Обратная совместимость: если значение уже int, оставляем.
        В реальном применении квантование должно сохранять масштаб, но для упрощения
        здесь мы приводим значения к диапазону 0-255 на основе min-max нормализации.
        """
        if not vector:
            return []

        # Простая проверка - если вектор уже состоит из интов, не квантуем повторно
        if all(isinstance(v, int) for v in vector):
            return vector

        min_v = min(vector)
        max_v = max(vector)

        if max_v == min_v:
            return [0] * len(vector)

        # Нормализуем в диапазон [0, 2^num_bits - 1]
        scale = (2**num_bits - 1) / (max_v - min_v)
        quantized = [int(round((v - min_v) * scale)) for v in vector]
        return quantized

    def _dequantize_vector(self, q_vector: List[int], min_v: float, max_v: float, num_bits: int = 8) -> List[float]:
        """Восстанавливает исходный вектор из квантованного."""
        if not q_vector:
            return []

        if max_v == min_v:
            return [min_v] * len(q_vector)

        scale = (max_v - min_v) / (2**num_bits - 1)
        dequantized = [min_v + (v * scale) for v in q_vector]
        return dequantized

    def add_item(self, text: str, vector: List[float], metadata: Optional[Dict[str, Any]] = None, domain: str = "general", quantize: bool = False):
        """Добавляет элемент в векторное хранилище."""
        if metadata is None:
            metadata = {}
        if "last_accessed" not in metadata:
            metadata["last_accessed"] = time.time()

        stored_vector = vector

        if quantize and vector:
            min_v = min(vector)
            max_v = max(vector)
            stored_vector = self._quantize_vector(vector)
            metadata["is_quantized"] = True
            metadata["quantize_min"] = min_v
            metadata["quantize_max"] = max_v

        item = {
            "domain": domain,
            "text": text,
            "vector": stored_vector,
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

    def _get_float_vector(self, item: Dict[str, Any]) -> List[float]:
        """Возвращает вектор в формате float. Декодирует если нужно."""
        vector = item.get("vector", [])
        metadata = item.get("metadata", {})

        if metadata.get("is_quantized", False):
            return self._dequantize_vector(
                vector,
                metadata.get("quantize_min", 0.0),
                metadata.get("quantize_max", 1.0)
            )
        return vector

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
                # Update cache item order for LRU
                self._search_cache.move_to_end(cache_key)
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
                float_vector = self._get_float_vector(item)
                base_score = self._cosine_similarity(query_vector, float_vector)

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
        # Apply LRU max_size limit
        if len(self._search_cache) > self.max_cache_size:
            self._search_cache.popitem(last=False)

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

    def dynamic_quantize(self, threshold: float = 2.0):
        """
        Проходит по всем элементам памяти. Если importance_weight меньше threshold
        и вектор еще не квантован, квантует его.
        Элементы с весом >= threshold остаются в float.
        """
        changed = False
        for item in self.memory:
            metadata = item.get("metadata", {})
            weight = metadata.get("importance_weight", 1.0)
            is_quantized = metadata.get("is_quantized", False)

            if weight < threshold and not is_quantized:
                vector = item.get("vector", [])
                if vector:
                    min_v = min(vector)
                    max_v = max(vector)
                    item["vector"] = self._quantize_vector(vector)
                    item["metadata"]["is_quantized"] = True
                    item["metadata"]["quantize_min"] = min_v
                    item["metadata"]["quantize_max"] = max_v
                    changed = True

        if changed:
            self._save()
