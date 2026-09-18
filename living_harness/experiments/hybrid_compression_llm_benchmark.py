import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import random
from typing import List, Dict, Any, Optional
from living_harness.core.hybrid_compressor import HybridCompressor
from living_harness.core.local_llm_connector import LocalLLMConnector

class MockLLMConnector(LocalLLMConnector):
    def generate(self, prompt: str, model: str = "mock", max_tokens: int = 1024, temperature: float = 0.7) -> Optional[str]:
        return "Это сгенерированное LLM резюме текста. Оно должно быть более осмысленным, чем просто первое предложение каждого блока. Ключевые поинты: искусственный интеллект, память, оптимизация."

def generate_mock_context(num_items: int) -> List[Dict[str, Any]]:
    context = []
    for i in range(num_items):
        item = {
            'text': f"Message {i}. This is a detailed explanation about topic {i%5}. We should consider the implications of this approach. It has several nuances that are crucial to understand.",
            'vector': [random.random() for _ in range(128)]
        }
        context.append(item)
    return context

def calculate_cosine_similarity(v1: List[float], v2: List[float]) -> float:
    t1 = torch.tensor(v1)
    t2 = torch.tensor(v2)
    if torch.norm(t1) == 0 or torch.norm(t2) == 0:
        return 0.0
    return torch.nn.functional.cosine_similarity(t1.unsqueeze(0), t2.unsqueeze(0)).item()

def run_benchmark():
    llm_connector = MockLLMConnector()

    compressor_heuristic = HybridCompressor(compression_ratio=0.5)
    compressor_llm = HybridCompressor(compression_ratio=0.5, llm_connector=llm_connector)

    num_items = 5
    context = generate_mock_context(num_items)

    print(f"Starting benchmark with {num_items} context items...")

    original_vectors = [item['vector'] for item in context]
    original_mean_vector = compressor_heuristic.aggregate_vectors(original_vectors)

    log_messages = []
    log_messages.append(f"Starting benchmark with {num_items} items.")
    log_messages.append("-" * 40)

    # Test heuristic
    compressed_item_h = compressor_heuristic.compress(context)
    sim_h = calculate_cosine_similarity(original_mean_vector, compressed_item_h['compressed_vector'])
    original_text_len = sum(len(c.get('text', '')) for c in context)

    msg_h = (f"Heuristic Compression:\n"
             f"Original text length vs Compressed text length: {original_text_len} -> {len(compressed_item_h['compressed_text'])}\n"
             f"Text snippet: {compressed_item_h['compressed_text'][:100]}...\n"
             f"Cosine Similarity of semantic vector to original baseline: {sim_h:.4f}")
    print(msg_h)
    log_messages.append(msg_h)

    log_messages.append("-" * 40)

    # Test LLM
    compressed_item_llm = compressor_llm.compress(context)
    sim_llm = calculate_cosine_similarity(original_mean_vector, compressed_item_llm['compressed_vector'])

    msg_llm = (f"LLM Compression:\n"
             f"Original text length vs Compressed text length: {original_text_len} -> {len(compressed_item_llm['compressed_text'])}\n"
             f"Text snippet: {compressed_item_llm['compressed_text'][:100]}...\n"
             f"Cosine Similarity of semantic vector to original baseline: {sim_llm:.4f}")
    print(msg_llm)
    log_messages.append(msg_llm)

    log_content = "\n\n".join(log_messages)

    os.makedirs('living_harness/data', exist_ok=True)
    with open('living_harness/data/hybrid_compression_llm_results.txt', 'w', encoding='utf-8') as f:
        f.write(log_content)

    print("Benchmark complete. Results saved to living_harness/data/hybrid_compression_llm_results.txt")

if __name__ == "__main__":
    run_benchmark()
