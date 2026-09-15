import json
import urllib.request
import urllib.error
from typing import Dict, Any, Optional

class LocalLLMConnector:
    """
    Коннектор для взаимодействия с локальным инференсом (например, через Ollama или vLLM).
    """
    def __init__(self, base_url: str = "http://localhost:11434", backend: str = "ollama"):
        self.base_url = base_url.rstrip('/')
        self.backend = backend.lower()
        if self.backend not in ["ollama", "vllm"]:
            raise ValueError("Supported backends are 'ollama' and 'vllm'")

    def generate(self, prompt: str, model: str = "llama3", max_tokens: int = 1024, temperature: float = 0.7) -> Optional[str]:
        if self.backend == "ollama":
            return self._generate_ollama(prompt, model, temperature)
        elif self.backend == "vllm":
            return self._generate_vllm(prompt, model, max_tokens, temperature)
        return None

    def _generate_ollama(self, prompt: str, model: str, temperature: float) -> Optional[str]:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature
            }
        }
        return self._make_request(url, payload, response_key="response")

    def _generate_vllm(self, prompt: str, model: str, max_tokens: int, temperature: float) -> Optional[str]:
        url = f"{self.base_url}/v1/completions"
        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        try:
            response_data = self._make_request_raw(url, payload)
            if response_data:
                choices = response_data.get("choices", [])
                if choices:
                    return choices[0].get("text", "")
            return ""
        except Exception as e:
            print(f"Error extracting vLLM response: {e}")
            return None

    def _make_request(self, url: str, payload: dict, response_key: str) -> Optional[str]:
        response_data = self._make_request_raw(url, payload)
        if response_data:
            return response_data.get(response_key, "")
        return None

    def _make_request_raw(self, url: str, payload: dict) -> Optional[dict]:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as e:
            print(f"Error connecting to {self.backend}: {e}")
            return None
