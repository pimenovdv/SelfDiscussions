import datetime
from typing import List, Dict, Any

class ContextManager:
    def __init__(self, system_prompt: str, max_window_tokens: int = 4096):
        """
        Управляет контекстом «живого ИИ», разделяя его на системный промпт, память и окно рассуждений.
        """
        self.system_prompt = system_prompt
        self.memory: List[Dict[str, str]] = []
        self.reasoning_window: List[Dict[str, Any]] = []
        self.max_window_tokens = max_window_tokens

    def add_memory(self, content: str) -> None:
        """Добавляет блок памяти с текущей меткой времени."""
        timestamp = datetime.datetime.now().isoformat()
        self.memory.append({"timestamp": timestamp, "content": content})

    def add_reasoning(self, text: str, estimated_tokens: int) -> None:
        """
        Добавляет новый блок рассуждений. Если окно заполнено на 95%,
        происходит усечение (оставляем только последние 10% лимита).
        """
        self.reasoning_window.append({"text": text, "tokens": estimated_tokens})
        self._truncate_if_needed()

    def _truncate_if_needed(self) -> None:
        """
        Логика усечения: при достижении 95% от max_window_tokens
        окно очищается от старых рассуждений так, чтобы остался объем не более 10% от лимита.
        """
        current_tokens = sum(item["tokens"] for item in self.reasoning_window)
        threshold_95 = 0.95 * self.max_window_tokens
        threshold_10 = 0.10 * self.max_window_tokens

        if current_tokens >= threshold_95:
            # Усекаем старые данные, пока текущий размер не станет <= 10% лимита
            while self.reasoning_window and current_tokens > threshold_10:
                removed_item = self.reasoning_window.pop(0)
                current_tokens -= removed_item["tokens"]

    def build_prompt(self) -> str:
        """Формирует итоговый контекст для модели."""
        prompt_parts = [
            "<|system|>",
            self.system_prompt,
            "\n<|memory|>"
        ]

        for mem in self.memory:
            prompt_parts.append(f"[{mem['timestamp']}] {mem['content']}")

        prompt_parts.append("\n<|reasoning|>")
        for reason in self.reasoning_window:
            prompt_parts.append(reason["text"])

        return "\n".join(prompt_parts)
