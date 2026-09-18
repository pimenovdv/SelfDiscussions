import re
from typing import List, Optional, Dict
from living_harness.core.agent import Agent
from living_harness.core.local_llm_connector import LocalLLMConnector

class AgentRouter:
    """
    Маршрутизатор входящих запросов, определяющий наиболее подходящего агента.
    Поддерживает LLM-классификатор и эвристический (keyword-based) запасной вариант.
    """
    def __init__(self, agents: List[Agent], llm_connector: Optional[LocalLLMConnector] = None):
        self.agents = {agent.name: agent for agent in agents}
        self.llm_connector = llm_connector

    def add_agent(self, agent: Agent):
        self.agents[agent.name] = agent

    def route(self, user_message: str) -> Optional[Agent]:
        if not self.agents:
            return None

        if len(self.agents) == 1:
            return list(self.agents.values())[0]

        if self.llm_connector:
            routed_agent = self._llm_route(user_message)
            if routed_agent:
                return routed_agent

        return self._heuristic_route(user_message)

    def _llm_route(self, user_message: str) -> Optional[Agent]:
        # Составляем описания агентов на основе их системных промптов (берем начало промпта для краткости)
        agent_descriptions = "\n".join([
            f"- {name}: {agent.context_manager.system_prompt.strip()[:150]}..."
            for name, agent in self.agents.items()
        ])

        prompt = (
            "You are a routing assistant. Your task is to select the most appropriate agent "
            "to answer the user's message based on their roles.\n"
            f"Available agents:\n{agent_descriptions}\n\n"
            f"User message: '{user_message}'\n\n"
            "Output ONLY the name of the selected agent."
        )

        response = self.llm_connector.generate(prompt, temperature=0.1)
        if response:
            selected = response.strip()
            # Пытаемся найти точное или частичное совпадение
            for name in self.agents:
                if name.lower() in selected.lower():
                    return self.agents[name]
        return None

    def _heuristic_route(self, user_message: str) -> Agent:
        """Fallback: эвристический поиск по пересечению слов (TF-like)."""
        msg_words = set(re.findall(r'\w+', user_message.lower()))
        best_agent = None
        max_score = -1

        for name, agent in self.agents.items():
            prompt_words = set(re.findall(r'\w+', agent.context_manager.system_prompt.lower()))
            score = len(msg_words.intersection(prompt_words))
            if score > max_score:
                max_score = score
                best_agent = agent

        # Если совпадений нет, берем первого попавшегося
        if max_score == 0 and self.agents:
            return list(self.agents.values())[0]

        return best_agent
