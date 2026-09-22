from living_harness.memory.vector_store import VectorStore
import datetime
from typing import List, Dict, Any
from living_harness.core.decay_mechanisms import RelevanceDecay
from living_harness.core.hybrid_compressor import HybridCompressor
from living_harness.core.structural_cohesion import calculate_structural_importance, merge_entropy_and_cohesion
from living_harness.analytics.entropy_analyzer import calculate_shannon_entropy


class ContextManager:
    def __init__(self, system_prompt: str, max_window_tokens: int = 4096, decay_rate: float = 0.05, enable_hybrid_compression: bool = True):
        self.decay_mechanism = RelevanceDecay(base_decay_rate=decay_rate)
        self.compressor = HybridCompressor() if enable_hybrid_compression else None
        self.vector_store = VectorStore()

        """
        Управляет контекстом «живого ИИ», разделяя его на системный промпт, память и окно рассуждений.
        """
        self.system_prompt = system_prompt
        self.memory: List[Dict[str, str]] = []
        self.reasoning_window: List[Dict[str, Any]] = []
        self.max_window_tokens = max_window_tokens
        self.enable_hybrid_compression = enable_hybrid_compression

    def add_memory(self, content: str) -> None:
        """Добавляет блок памяти с текущей меткой времени."""
        timestamp = datetime.datetime.now().isoformat()
        self.memory.append({"timestamp": timestamp, "content": content})

    def add_reasoning(self, text: str, estimated_tokens: int, vector: List[float] = None) -> None:
        """
        Добавляет новый блок рассуждений. Если окно заполнено на 95%,
        происходит усечение (оставляем только последние 10% лимита) или компрессия.
        """
        self.reasoning_window.append({"text": text, "tokens": estimated_tokens, "vector": vector})
        self._truncate_if_needed()

    def _apply_decay_and_prune_memory(self) -> None:
        """
        Применяет морфологическое увядание к памяти. Записи, чья релевантность
        падает ниже порога, удаляются.
        """
        current_time = datetime.datetime.now()
        threshold = 0.1 # Порог релевантности для удаления

        # Идем с конца, чтобы безопасно удалять элементы по индексу
        for i in range(len(self.memory) - 1, -1, -1):
            mem = self.memory[i]
            if "timestamp" in mem and mem["timestamp"]:
                try:
                    mem_time = datetime.datetime.fromisoformat(mem["timestamp"])
                    age_minutes = (current_time - mem_time).total_seconds() / 60.0
                    relevance = self.decay_mechanism.calculate_penalty(age_minutes)

                    if relevance < threshold:
                        self.memory.pop(i)
                except ValueError:
                    pass

    def _truncate_if_needed(self) -> None:
        """
        Логика усечения: при достижении 95% от max_window_tokens
        окно очищается на основе метрик структурной связности и локальной энтропии.
        Вытесняются элементы с наименьшей комбинированной важностью, пока объем не станет <= 10% от лимита,
        либо применяется гибридная компрессия.
        """
        current_tokens = sum(item["tokens"] for item in self.reasoning_window)
        threshold_95 = 0.95 * self.max_window_tokens
        threshold_10 = 0.10 * self.max_window_tokens

        if current_tokens >= threshold_95:
            # Вычисляем важность каждого элемента окна
            entropies = []
            cohesions = []
            for item in self.reasoning_window:
                text = item["text"]
                entropy = calculate_shannon_entropy(text)

                # Структурная связность: усредняем важность токенов
                token_importances = calculate_structural_importance(text)
                if token_importances:
                    cohesion = sum(token_importances) / len(token_importances)
                else:
                    cohesion = 0.0

                entropies.append(entropy)
                cohesions.append(cohesion)

            # Нормализация энтропии (чтобы привести к масштабу 0-1 для комбинации)
            max_entropy = max(entropies) if entropies and max(entropies) > 0 else 1.0
            normalized_entropies = [e / max_entropy for e in entropies]

            # Комбинируем метрики
            retention_scores = merge_entropy_and_cohesion(normalized_entropies, cohesions, alpha=0.5)

            # Прикрепляем оценки к индексам и сортируем по возрастанию оценки (наименее важные в начале)
            scored_items = sorted(enumerate(retention_scores), key=lambda x: x[1])

            items_to_remove_indices = []
            tokens_to_remove = current_tokens - threshold_10
            removed_tokens_count = 0

            for idx, score in scored_items:
                if removed_tokens_count >= tokens_to_remove:
                    break
                items_to_remove_indices.append(idx)
                removed_tokens_count += self.reasoning_window[idx]["tokens"]

            # Сортируем индексы по убыванию, чтобы безопасно удалять из списка
            items_to_remove_indices.sort(reverse=True)

            if self.enable_hybrid_compression and self.compressor and len(items_to_remove_indices) > 0:
                items_to_compress = []
# Извлечение элементов в хронологическом порядке
                items_to_remove_indices.sort() # Сортируем по возрастанию для извлечения в правильном порядке

                # Удаляем с конца, чтобы не сбить индексы, но сохраняем в правильном порядке
                # Проще создать новый список для reasoning_window
                new_reasoning_window = []
                for i, item in enumerate(self.reasoning_window):
                    if i in items_to_remove_indices:
                        items_to_compress.append(item)
                    else:
                        new_reasoning_window.append(item)

                self.reasoning_window = new_reasoning_window

                if items_to_compress:
                    compressed = self.compressor.compress(items_to_compress)
                    compressed_text = "[СЖАТОЕ ПРОШЛОЕ РАССУЖДЕНИЕ] " + compressed['compressed_text']

                    if compressed.get('compressed_vector'):
                        self.vector_store.add_item(compressed['compressed_text'], compressed['compressed_vector'])

                    compressed_item = {
                        "text": compressed_text,
                        "tokens": len(compressed_text) // 4,
                        "vector": compressed['compressed_vector']
                    }
                    self.reasoning_window.insert(0, compressed_item)
            else:
                # Просто удаляем элементы с наименьшей оценкой
                for idx in items_to_remove_indices:
                    self.reasoning_window.pop(idx)

    def build_prompt(self) -> str:
        """Формирует итоговый контекст для модели с учетом увядания памяти."""
        self._apply_decay_and_prune_memory()
        prompt_parts = [
            "<|system|>",
            self.system_prompt,
            "\n<|memory|>"
        ]

        for mem in self.memory:
            if "timestamp" in mem and mem["timestamp"]:
                prompt_parts.append(f"[{mem['timestamp']}] {mem['content']}")
            else:
                prompt_parts.append(f"{mem.get('content', '')}")

        prompt_parts.append("\n<|reasoning|>")
        for reason in self.reasoning_window:
            prompt_parts.append(reason["text"])

        return "\n".join(prompt_parts)

    def retrieve_relevant_memory(self, query_vector: List[float], top_k: int = 1) -> str:
        """Извлекает релевантную долговременную память по вектору."""
        results = self.vector_store.search(query_vector, top_k)
        if results:
            return "\n".join([f"[ИЗВЛЕЧЕНО ИЗ ПАМЯТИ] {item['text']}" for item in results])
        return ""
