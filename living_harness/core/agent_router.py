import re
from typing import List, Optional, Dict
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from living_harness.core.agent import Agent
from living_harness.core.local_llm_connector import LocalLLMConnector

class AgentRouter:
    """
    Маршрутизатор входящих запросов, определяющий наиболее подходящего агента.
    Поддерживает LLM-классификатор, векторный поиск по эмбеддингам и эвристический (keyword-based) запасной вариант.
    """
    def __init__(self, agents: List[Agent], llm_connector: Optional[LocalLLMConnector] = None, use_embeddings: bool = False):
        self.agents = {agent.name: agent for agent in agents}
        self.llm_connector = llm_connector
        self.use_embeddings = use_embeddings

        self.tokenizer = None
        self.model = None
        self.agent_embeddings = {}

        if self.use_embeddings:
            # Инициализируем легковесную модель эмбеддингов
            # Для скорости можно использовать sentence-transformers/all-MiniLM-L6-v2
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModel.from_pretrained(model_name)
                self.model.eval()
                # Кэшируем эмбеддинги системных промптов агентов
                self._cache_agent_embeddings()
            except Exception as e:
                print(f"Warning: Failed to load embedding model: {e}")
                self.use_embeddings = False

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def _get_embedding(self, text: str) -> torch.Tensor:
        if not self.tokenizer or not self.model:
            return torch.zeros(1, 384) # fallback

        encoded_input = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        sentence_embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
        return F.normalize(sentence_embeddings, p=2, dim=1)

    def _cache_agent_embeddings(self):
        for name, agent in self.agents.items():
            self.agent_embeddings[name] = self._get_embedding(agent.context_manager.system_prompt)

    def add_agent(self, agent: Agent):
        self.agents[agent.name] = agent
        if self.use_embeddings:
            self.agent_embeddings[agent.name] = self._get_embedding(agent.context_manager.system_prompt)

    def route(self, user_message: str) -> Optional[Agent]:
        if not self.agents:
            return None

        if len(self.agents) == 1:
            return list(self.agents.values())[0]

        if self.use_embeddings:
            routed_agent = self._embedding_route(user_message)
            if routed_agent:
                return routed_agent

        if self.llm_connector:
            routed_agent = self._llm_route(user_message)
            if routed_agent:
                return routed_agent

        return self._heuristic_route(user_message)

    def _embedding_route(self, user_message: str) -> Optional[Agent]:
        """
        Маршрутизация на основе векторного сходства запроса и системных промптов.
        """
        request_tensor = self._get_embedding(user_message)

        best_agent = None
        max_sim = -1.0

        for name, agent in self.agents.items():
            agent_tensor = self.agent_embeddings.get(name)
            if agent_tensor is None:
                continue

            sim = F.cosine_similarity(request_tensor, agent_tensor).item()
            if sim > max_sim:
                max_sim = sim
                best_agent = agent

        return best_agent

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
