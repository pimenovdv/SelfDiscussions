import re
import json
import os
import math
import time
from collections import OrderedDict
from sklearn.cluster import KMeans
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple

class VectorStore:
    def identify_curiosity_targets(self) -> List[str]:
        nodes_set = set()
        in_degrees = {}
        out_degrees = {}

        for item in self.memory:
            domain = item.get("domain", "general")
            nodes_set.add(domain)

        for node in nodes_set:
            in_degrees[node] = 0
            out_degrees[node] = 0

        for source, targets in self.transitions.items():
            nodes_set.add(source)
            for target, weight in targets.items():
                nodes_set.add(target)
                out_degrees[source] = out_degrees.get(source, 0) + weight
                in_degrees[target] = in_degrees.get(target, 0) + weight

        curiosity_targets = []
        for node in nodes_set:
            total_degree = in_degrees.get(node, 0) + out_degrees.get(node, 0)
            if total_degree < 2:
                curiosity_targets.append(node)

        return curiosity_targets


    def generate_curiosity_hypothesis(self) -> List[str]:
        """
        Генерирует гипотезы для изолированных доменов, выявленных через identify_curiosity_targets.
        """
        targets = self.identify_curiosity_targets()
        hypotheses = []
        for target in targets:
            hypotheses.append(f"Hypothesis: Domain '{target}' might be connected to other domains.")
        return hypotheses

    def validate_hypothesis_dynamic(self, hypothesis_text: str, hypothesis_vector: List[float], domain: str, base_threshold: float = 0.5) -> bool:
        """
        Оценивает правдоподобность гипотезы с адаптивным порогом, зависящим от энтропии домена.
        """
        entropy = self.calculate_domain_entropy(domain)
        # Если энтропия высока (разнообразие), порог снижается (мягче), если низка - повышается (строже)
        # Коэффициент 0.1 - для тонкой настройки
        dynamic_threshold = max(0.1, min(0.9, base_threshold - (entropy * 0.1)))

        results = self.search(query_vector=hypothesis_vector, top_k=3, domain=domain)
        if not results:
            return False

        avg_score = sum(self._cosine_similarity(hypothesis_vector, self._get_float_vector(item)) for item in results) / len(results)
        return avg_score >= dynamic_threshold

    def validate_hypothesis(self, hypothesis_text: str, hypothesis_vector: List[float], threshold: float = 0.5) -> bool:
        """
        Оценивает правдоподобность гипотезы на основе векторной близости к существующим знаниям.
        Если среднее косинусное сходство с top-3 ближайшими соседями превышает порог, гипотеза принимается.
        """
        results = self.search(query_vector=hypothesis_vector, top_k=3)
        if not results:
            return False

        avg_score = sum(self._cosine_similarity(hypothesis_vector, self._get_float_vector(item)) for item in results) / len(results)
        return avg_score >= threshold
    def __init__(self, db_path: str = "living_harness/data/vector_memory.json", max_cache_size: int = 100):
        self.db_path = db_path
        self.max_cache_size = max_cache_size
        self.memory: List[Dict[str, Any]] = []
        # Cache format: query_tuple -> (timestamp, result_list)
        # Using OrderedDict to implement LRU Cache
        self._search_cache: OrderedDict[Tuple[Any, ...], Tuple[float, List[Dict[str, Any]]]] = OrderedDict()
        self._cache_ttl = 60.0 # seconds
        self.transitions: Dict[str, Dict[str, int]] = {}
        self.last_query_domain: Optional[str] = None
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

    def add_item(self, text: str, vector: List[float], metadata: Optional[Dict[str, Any]] = None, domain: str = "general", quantize: bool = False, emotion: Optional[str] = None):
        """Добавляет элемент в векторное хранилище."""
        if metadata is None:
            metadata = {}
        if "last_accessed" not in metadata:
            metadata["last_accessed"] = time.time()
        if emotion:
            metadata["emotion"] = emotion

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

    def search(self, query_vector: List[float], top_k: int = 3, domain: Optional[Union[str, List[str]]] = None, emotional_state: Optional[str] = None) -> List[Dict[str, Any]]:
        """Ищет top_k наиболее похожих элементов по косинусному сходству."""

        # Track domain transitions for proactive prediction
        current_domain = None
        if isinstance(domain, str):
            current_domain = domain
        elif isinstance(domain, list) and len(domain) > 0:
            current_domain = domain[0] # Simplification

        if self.last_query_domain and current_domain:
            if self.last_query_domain not in self.transitions:
                self.transitions[self.last_query_domain] = {}
            self.transitions[self.last_query_domain][current_domain] = self.transitions[self.last_query_domain].get(current_domain, 0) + 1

        self.last_query_domain = current_domain

        # Serialize domain for cache key
        if domain is None:
            domain_key = None
        elif isinstance(domain, str):
            domain_key = domain
        else:
            domain_key = tuple(sorted(domain))

        cache_key = (tuple(query_vector), top_k, domain_key, emotional_state)
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

                if emotional_state and emotional_state == item.get("metadata", {}).get("emotion"):
                    final_score *= 1.2 # Эмоциональный резонанс

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


    def cluster_and_abstract(self, n_clusters: int = 3):
        """
        Группирует существующие в памяти (не квантованные) векторы
        и создает новые элементы - 'абстракции' (центроиды кластеров).
        """
        float_vectors = []
        indices = []
        for i, item in enumerate(self.memory):
            meta = item.get("metadata", {})
            if not meta.get("is_quantized", False) and not meta.get("is_abstract", False):
                vec = item.get("vector")
                if vec:
                    float_vectors.append(vec)
                    indices.append(i)

        if len(float_vectors) < n_clusters:
            return # Недостаточно данных для кластеризации

        X = np.array(float_vectors)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto").fit(X)
        centroids = kmeans.cluster_centers_

        for i, centroid in enumerate(centroids):
            abstract_text = f"Abstract Concept: cluster_{i}"
            # Проверяем, нет ли уже такого абстрактного концепта, если есть, обновляем
            existing_item = None
            for item in self.memory:
                if item.get("text") == abstract_text and item.get("metadata", {}).get("is_abstract"):
                    existing_item = item
                    break

            if existing_item:
                existing_item["vector"] = centroid.tolist()
                self._save()
            else:
                self.add_item(
                    text=abstract_text,
                    vector=centroid.tolist(),
                    metadata={"is_abstract": True, "cluster_id": i},
                    domain="abstract"
                )


    def export_domain_graph(self) -> Dict[str, Any]:
        """
        Экспортирует матрицу переходов между доменами в формате графа
        (node-link data structure, совместимом с NetworkX).
        """
        nodes_set = set()
        for source, targets in self.transitions.items():
            nodes_set.add(source)
            for target in targets.keys():
                nodes_set.add(target)

        nodes = [{"id": node} for node in nodes_set]
        links = []
        for source, targets in self.transitions.items():
            for target, weight in targets.items():
                links.append({"source": source, "target": target, "weight": weight})

        return {
            "directed": True,
            "multigraph": False,
            "graph": {},
            "nodes": nodes,
            "edges": links
        }

    def predict_next_domain(self, current_domain: str) -> Optional[str]:
        """Предсказывает наиболее вероятный следующий домен запроса на основе истории переходов."""
        if current_domain not in self.transitions or not self.transitions[current_domain]:
            return None

        # Find the domain with the highest transition count
        next_domain = max(self.transitions[current_domain].items(), key=lambda x: x[1])[0]
        return next_domain

    def prefetch_predicted_domain(self, current_domain: str, top_k: int = 5):
        """
        Предзагружает данные предсказанного домена в кэш.
        """
        next_domain = self.predict_next_domain(current_domain)
        if not next_domain:
            return

        # Для прогрева кэша выполняем поиск с "нулевым" вектором
        dummy_vector = []
        for item in self.memory:
            if item.get("domain") == next_domain and item.get("vector"):
                dummy_vector = [0.0] * len(item["vector"])
                break

        if dummy_vector:
            self.search(query_vector=dummy_vector, top_k=top_k, domain=next_domain)


    def prune_stale_abstractions(self, threshold: float = 1.05):
        """
        Удаляет абстрактные концепты (is_abstract=True), чей importance_weight упал ниже порога.
        Это симулирует забывание устаревших или бесполезных обобщений.
        """
        initial_count = len(self.memory)
        self.memory = [
            item for item in self.memory
            if not (item.get("metadata", {}).get("is_abstract", False) and
                    item.get("metadata", {}).get("importance_weight", 1.0) < threshold)
        ]

        if len(self.memory) < initial_count:
            self._save()

    def consolidate_memory(self, similarity_threshold: float = 0.85):
        """
        Объединяет схожие абстрактные концепты в один. Если косинусное расстояние между
        двумя абстракциями больше similarity_threshold, они сливаются: их векторы
        усредняются, веса складываются, а старые узлы удаляются.
        """
        abstract_items = [
            (i, item) for i, item in enumerate(self.memory)
            if item.get("metadata", {}).get("is_abstract", False)
        ]

        if len(abstract_items) < 2:
            return

        to_remove = set()
        new_items = []

        # Простой O(N^2) поиск пар для консолидации (можно оптимизировать)
        for idx1, (mem_idx1, item1) in enumerate(abstract_items):
            if mem_idx1 in to_remove:
                continue

            for idx2 in range(idx1 + 1, len(abstract_items)):
                mem_idx2, item2 = abstract_items[idx2]
                if mem_idx2 in to_remove:
                    continue

                vec1 = self._get_float_vector(item1)
                vec2 = self._get_float_vector(item2)

                if not vec1 or not vec2:
                    continue

                sim = self._cosine_similarity(vec1, vec2)

                if sim >= similarity_threshold:
                    # Консолидация!
                    to_remove.add(mem_idx1)
                    to_remove.add(mem_idx2)

                    # Усредняем векторы
                    new_vec = [(v1 + v2) / 2.0 for v1, v2 in zip(vec1, vec2)]

                    # Суммируем веса, но ограничиваем максимумом (например, 10.0)
                    weight1 = item1.get("metadata", {}).get("importance_weight", 1.0)
                    weight2 = item2.get("metadata", {}).get("importance_weight", 1.0)
                    new_weight = min(weight1 + weight2, 10.0)

                    new_item = {
                        "text": f"Consolidated Concept: {item1.get('text')} + {item2.get('text')}",
                        "vector": new_vec,
                        "metadata": {
                            "is_abstract": True,
                            "importance_weight": new_weight,
                            "consolidated_from": [item1.get("text"), item2.get("text")]
                        },
                        "domain": "abstract"
                    }
                    new_items.append(new_item)
                    break # item1 уже слит, переходим к следующему

        if to_remove or new_items:
            # Обновляем память
            self.memory = [item for i, item in enumerate(self.memory) if i not in to_remove]
            self.memory.extend(new_items)
            self._save()

    def reflect_on_monologue(self, similarity_threshold: float = 0.8) -> None:
        """
        Ищет паттерны (схожие векторы) среди элементов типа 'internal_monologue'
        и формирует 'meta_knowledge' вывод.
        """
        monologues = [
            (i, item) for i, item in enumerate(self.memory)
            if item.get("metadata", {}).get("type") == "internal_monologue"
        ]

        if len(monologues) < 2:
            return

        to_remove = set()
        new_items = []

        for idx1, (mem_idx1, item1) in enumerate(monologues):
            if mem_idx1 in to_remove:
                continue

            for idx2 in range(idx1 + 1, len(monologues)):
                mem_idx2, item2 = monologues[idx2]
                if mem_idx2 in to_remove:
                    continue

                vec1 = self._get_float_vector(item1)
                vec2 = self._get_float_vector(item2)

                if not vec1 or not vec2:
                    continue

                sim = self._cosine_similarity(vec1, vec2)
                if sim >= similarity_threshold:
                    to_remove.add(mem_idx1)
                    to_remove.add(mem_idx2)

                    new_vec = [(v1 + v2) / 2.0 for v1, v2 in zip(vec1, vec2)]

                    new_item = {
                        "text": f"Meta-Reflection: Found pattern in thoughts: {item1.get('text')} & {item2.get('text')}",
                        "vector": new_vec,
                        "metadata": {
                            "type": "meta_knowledge",
                            "importance_weight": 2.0
                        },
                        "domain": "abstract"
                    }
                    new_items.append(new_item)
                    break

        if to_remove or new_items:
            self.memory = [item for i, item in enumerate(self.memory) if i not in to_remove]
            self.memory.extend(new_items)
            self._save()

    def calculate_domain_entropy(self, domain: str) -> float:
        """
        Оценивает дисперсию (разброс) векторов внутри домена как прокси-метрику информационной энтропии.
        Если домен содержит менее 2 элементов, энтропия считается 0.0.
        """
        float_vectors = []
        for item in self.memory:
            if item.get("domain") == domain:
                vec = self._get_float_vector(item)
                if vec:
                    float_vectors.append(vec)

        if len(float_vectors) < 2:
            return 0.0

        X = np.array(float_vectors)
        # Calculate variance along each dimension, then take the mean variance across dimensions
        variances = np.var(X, axis=0)
        mean_variance = float(np.mean(variances))

        return mean_variance

    def forget_old_memories(self, threshold: float = 0.5):
        """
        Удаляет обычные (не абстрактные и не мета) элементы памяти, чей importance_weight упал ниже порога.
        Имитирует процесс забывания редко используемой информации.
        """
        initial_count = len(self.memory)
        self.memory = [
            item for item in self.memory
            if item.get("metadata", {}).get("is_abstract", False) or
               item.get("metadata", {}).get("type") == "meta_knowledge" or
               item.get("metadata", {}).get("importance_weight", 1.0) >= threshold
        ]

        if len(self.memory) < initial_count:
            self._save()

    def calculate_boredom(self, recent_k: int = 10) -> float:
        """
        Оценивает уровень "скуки" (boredom signal) на основе семантического однообразия последних добавленных элементов.
        Возвращает значение от 0.0 (максимальная новизна) до 1.0 (максимальная скука/однообразие).
        """
        if len(self.memory) < recent_k:
            return 0.0

        recent_items = self.memory[-recent_k:]
        import numpy as np
        similarities = []

        for i in range(len(recent_items)):
            vec1 = self._get_float_vector(recent_items[i])
            if not vec1: continue
            for j in range(i + 1, len(recent_items)):
                vec2 = self._get_float_vector(recent_items[j])
                if not vec2: continue
                sim = self._cosine_similarity(vec1, vec2)
                similarities.append(sim)

        if not similarities:
            return 0.0

        return float(np.mean(similarities))

    def integrate_hypothesis_pipeline(self, generation_callback=None, vectorization_callback=None) -> List[dict]:
        """
        Объединяет механизмы любопытства, генерации и валидации в единый цикл.
        Находит изолированные домены, генерирует для них гипотезы, проверяет их,
        и сохраняет успешные в память как мета-знания.
        """
        integrated_hypotheses = []

        # generate_curiosity_hypothesis already calls identify_curiosity_targets() inside
        hypotheses = self.generate_curiosity_hypothesis()

        for hyp_text in hypotheses:
            # Extract target domain from hypothesis string "Hypothesis: Domain 'target' might be..."
            match = re.search(r"Domain '(.+?)'", hyp_text)
            if not match:
                continue
            target = match.group(1)

            if generation_callback:
                actual_hypothesis = generation_callback(target)
            else:
                actual_hypothesis = hyp_text

            if vectorization_callback:
                mock_vector = vectorization_callback(actual_hypothesis)
            else:
                # Fallback to zero vector if no model provided
                mock_vector = [0.0] * 384

            if self.validate_hypothesis_dynamic(actual_hypothesis, mock_vector, target):
                new_item = {
                    "text": actual_hypothesis,
                    "vector": mock_vector,
                    "domain": "meta",
                    "metadata": {
                        "type": "meta_knowledge",
                        "source_domain": target,
                        "importance_weight": 2.0
                    }
                }
                self.memory.append(new_item)
                integrated_hypotheses.append(new_item)

        if integrated_hypotheses:
            self._save()

        return integrated_hypotheses


    def spontaneous_activation(self, threshold: float = 0.6, boost_amount: float = 0.2):
        """
        Имитирует 'спонтанную активацию' памяти.
        Находит элементы, чей вес находится чуть выше порога забывания (риск быть забытыми),
        и слегка повышает их важность.
        """
        activated = False
        for item in self.memory:
            if not item.get("metadata", {}).get("is_abstract", False) and item.get("metadata", {}).get("type") != "meta_knowledge":
                current_weight = item.get("metadata", {}).get("importance_weight", 1.0)
                if threshold <= current_weight < (threshold + 0.3):
                    item["metadata"]["importance_weight"] = current_weight + boost_amount
                    activated = True

        if activated:
            self._save()


    async def background_curiosity_cycle(self, generation_callback=None, vectorization_callback=None, interval_seconds: float = 60.0):
        """Асинхронный фоновый цикл, который периодически ищет белые пятна, генерирует и валидирует гипотезы."""
        import asyncio
        while True:
            try:
                # Находим пробелы и генерируем гипотезы
                successful_hypotheses = self.integrate_hypothesis_pipeline(
                    generation_callback=generation_callback,
                    vectorization_callback=vectorization_callback
                )

                # Дополнительная логика логирования или уведомления могла бы быть здесь

            except Exception as e:
                print(f"Error in background_curiosity_cycle: {e}")

            await asyncio.sleep(interval_seconds)
