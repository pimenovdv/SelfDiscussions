import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import random
from typing import List, Dict, Any
from living_harness.core.hybrid_compressor import HybridCompressor

def generate_mock_context(num_items: int) -> List[Dict[str, Any]]:
    context = []
    for i in range(num_items):
        item = {
            'text': f"Message {i}. This is a detailed explanation about topic {i%5}. We should consider the implications of this approach.",
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
    compressor = HybridCompressor(compression_ratio=0.5)

    num_items = 100
    context = generate_mock_context(num_items)

    print(f"Starting benchmark with {num_items} context items...")

    original_vectors = [item['vector'] for item in context]
    original_mean_vector = compressor.aggregate_vectors(original_vectors)

    cycles = 5
    current_context = context.copy()

    log_messages = []
    log_messages.append(f"Starting benchmark with {num_items} items.")
    log_messages.append("-" * 40)

    for cycle in range(1, cycles + 1):
        compressed_item = compressor.compress(current_context)
        sim = calculate_cosine_similarity(original_mean_vector, compressed_item['compressed_vector'])

        new_items = generate_mock_context(10)

        repacked_compressed = {
            'text': compressed_item['compressed_text'],
            'vector': compressed_item['compressed_vector']
        }

        current_context = [repacked_compressed] + new_items

        original_text_len = sum(len(c.get('text', '')) for c in context)
        msg = (f"Cycle {cycle}: Compressed {compressed_item['original_count']} items into 1.\n"
               f"Original text length vs Compressed text length: {original_text_len} -> {len(compressed_item['compressed_text'])}\n"
               f"Cosine Similarity of semantic vector to original baseline: {sim:.4f}")
        print(msg)
        log_messages.append(msg)

    log_content = "\n\n".join(log_messages)

    os.makedirs('living_harness/data', exist_ok=True)
    with open('living_harness/data/hybrid_compression_results.txt', 'w', encoding='utf-8') as f:
        f.write(log_content)

    print("Benchmark complete. Results saved to living_harness/data/hybrid_compression_results.txt")

if __name__ == "__main__":
    run_benchmark()
