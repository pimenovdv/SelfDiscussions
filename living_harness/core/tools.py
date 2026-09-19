import datetime
from living_harness.core.semantic_checksum import SemanticChecksum
import time
from typing import Dict, Any

class Tools:
    def __init__(self, context_manager):
        """
        Инициализация инструментов взаимодействия для агента.

        :param context_manager: Экземпляр ContextManager для управления памятью.
        """
        self.context_manager = context_manager

    def ask_user(self, question: str) -> str:
        """
        Задает вопрос пользователю и возвращает его ответ.
        Используется для запроса уточнений или помощи.
        """
        print(f"\n[AI Agent спрашивает]: {question}")
        try:
            answer = input("Ответ: ")
            return answer
        except EOFError:
            return ""

    def new_memory(self, content: str) -> str:
        """
        Записывает новый блок информации в долгосрочную память.
        """
        self.context_manager.add_memory(content)
        return "Запись в память успешно добавлена."

    def overwrite_memory(self, index: int, new_content: str) -> str:
        """
        Перезаписывает существующий блок памяти по индексу с семантической проверкой.
        """
        if 0 <= index < len(self.context_manager.memory):
            old_content = self.context_manager.memory[index]["content"]
            if SemanticChecksum.verify_overwrite(old_content, new_content):
                self.context_manager.memory[index]["content"] = new_content
                self.context_manager.memory[index]["timestamp"] = datetime.datetime.now().isoformat()
                return f"Память по индексу {index} успешно перезаписана."
            else:
                return f"Ошибка: Семантический сдвиг слишком велик. Перезапись отклонена для предотвращения деградации памяти."
        else:
            return f"Ошибка: Индекс памяти {index} вне диапазона."

class HarnessTools:
    def __init__(self, context_manager, websocket=None):
        self.context_manager = context_manager
        self.websocket = websocket

    def new_memory(self, key: str, value: str) -> str:
        space = self.context_manager.get_memory_space()
        if len(self.context_manager.memories) >= space:
            return "Error: Memory full."

        self.context_manager.memories[key] = {
            "timestamp": time.time(),
            "data": value
        }
        # Update dynamic t
        estimated_tokens = len(value.split()) # simple heuristic
        self.context_manager.update_dynamic_t(estimated_tokens)
        return f"Memory '{key}' saved."

    def overwrite_memory(self, key: str, value: str) -> str:
        if key in self.context_manager.memories:
             self.context_manager.memories[key] = {
                 "timestamp": time.time(),
                 "data": value
             }
             estimated_tokens = len(value.split())
             self.context_manager.update_dynamic_t(estimated_tokens)
             return f"Memory '{key}' overwritten."
        return self.new_memory(key, value)

    def get_memory_space(self) -> int:
        return self.context_manager.get_memory_space()

    async def ask_user(self, message: str) -> str:
        print(f"\n[AI ASKS USER]: {message}")
        if self.websocket:
            await self.websocket.send_text(f"[AI]: {message}")
        return "Message sent to user."
