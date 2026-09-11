import datetime

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
        Перезаписывает существующий блок памяти по индексу.
        """
        if 0 <= index < len(self.context_manager.memory):
            self.context_manager.memory[index]["content"] = new_content
            self.context_manager.memory[index]["timestamp"] = datetime.datetime.now().isoformat()
            return f"Память по индексу {index} успешно перезаписана."
        else:
            return f"Ошибка: Индекс памяти {index} вне диапазона."
