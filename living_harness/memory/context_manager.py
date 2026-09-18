import datetime
from typing import List, Dict, Any
from living_harness.core.decay_mechanisms import RelevanceDecay
from living_harness.core.hybrid_compressor import HybridCompressor

class ContextManager:
    def __init__(self, system_prompt: str, max_window_tokens: int = 4096, decay_rate: float = 0.05, enable_hybrid_compression: bool = True):
        self.decay_mechanism = RelevanceDecay(base_decay_rate=decay_rate)
        self.compressor = HybridCompressor() if enable_hybrid_compression else None

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
        окно очищается от старых рассуждений так, чтобы остался объем не более 10% от лимита,
        либо применяется гибридная компрессия, если включена.
        """
        current_tokens = sum(item["tokens"] for item in self.reasoning_window)
        threshold_95 = 0.95 * self.max_window_tokens
        threshold_10 = 0.10 * self.max_window_tokens

        if current_tokens >= threshold_95:
            if self.enable_hybrid_compression and self.compressor and len(self.reasoning_window) > 1:
                items_to_compress = []
                while self.reasoning_window and current_tokens > threshold_10:
                    removed_item = self.reasoning_window.pop(0)
                    items_to_compress.append(removed_item)
                    current_tokens -= removed_item["tokens"]

                if items_to_compress:
                    compressed = self.compressor.compress(items_to_compress)
                    compressed_text = "[СЖАТОЕ ПРОШЛОЕ РАССУЖДЕНИЕ] " + compressed['compressed_text']

                    compressed_item = {
                        "text": compressed_text,
                        "tokens": len(compressed_text) // 4,
                        "vector": compressed['compressed_vector']
                    }
                    self.reasoning_window.insert(0, compressed_item)
            else:
                # Усекаем старые данные, пока текущий размер не станет <= 10% лимита
                while self.reasoning_window and current_tokens > threshold_10:
                    removed_item = self.reasoning_window.pop(0)
                    current_tokens -= removed_item["tokens"]

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
