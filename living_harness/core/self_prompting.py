import time
from living_harness.memory.vector_store import VectorStore

class SelfPrompter:
    def __init__(self, llm_connector=None, idle_threshold: float = 60.0, vector_store=None):
        self.llm_connector = llm_connector
        self.idle_threshold = idle_threshold
        self.last_active_time = time.time()
        self.vector_store = vector_store or VectorStore()

    def update_activity(self):
        """Обновляет время последней активности."""
        self.last_active_time = time.time()

    def should_self_prompt(self) -> bool:
        """Проверяет, превышено ли время бездействия."""
        return (time.time() - self.last_active_time) > self.idle_threshold

    def generate_goal(self, context_summary: str) -> str:
        """
        Генерирует новую автономную цель на основе текущего контекста.
        """
        if self.llm_connector:
            prompt = f"System: Based on the following context, suggest a new internal goal to explore:\n{context_summary}\nNew Goal:"
            goal = self.llm_connector.generate(prompt)

            # Сохраняем мысль, если она была сгенерирована
            if hasattr(self.llm_connector, 'last_thought') and self.llm_connector.last_thought:
                # Вектор должен генерироваться из текста мысли (здесь заглушка)
                dummy_vector = [0.1] * 384
                self.vector_store.add_item(
                    text=self.llm_connector.last_thought,
                    vector=dummy_vector,
                    metadata={"type": "internal_monologue", "timestamp": time.time()}
                )

            return goal.strip() if goal else "Explore abstract concepts."
        return f"Analyze recent memory patterns related to '{context_summary[:15]}...'"
