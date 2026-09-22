import time

class SelfPrompter:
    def __init__(self, llm_connector=None, idle_threshold: float = 60.0):
        self.llm_connector = llm_connector
        self.idle_threshold = idle_threshold
        self.last_active_time = time.time()

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
            return goal.strip() if goal else "Explore abstract concepts."
        return f"Analyze recent memory patterns related to '{context_summary[:15]}...'"
