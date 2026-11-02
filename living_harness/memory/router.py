from typing import List, Dict, Any
from living_harness.memory.vector_store import VectorStore

class AdaptiveRouter:
    """
    Адаптивный маршрутизатор для памяти.
    Использует VectorStore для предварительной маршрутизации запросов, определяя релевантные домены.
    """
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store

    def route(self, query_vector: List[float], base_similarity_threshold: float = 0.5, use_dynamic_threshold: bool = True) -> List[Dict[str, Any]]:
        """
        Использует contextual_routing для фильтрации элементов памяти по доменам,
        затем возвращает отфильтрованные элементы, сокращая пространство поиска.
        """
        return self.vector_store.contextual_routing(
            query_vector,
            base_similarity_threshold=base_similarity_threshold,
            use_dynamic_threshold=use_dynamic_threshold
        )
