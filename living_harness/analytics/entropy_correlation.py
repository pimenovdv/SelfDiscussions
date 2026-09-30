import os
import json
import math
from collections import Counter
import re
from typing import List, Dict

def calculate_shannon_entropy(text: str, window_size: int = 5) -> float:
    """Calculates Shannon entropy of n-grams (words) in a text."""
    words = re.findall(r'\b\w+\b', text.lower())
    if not words:
        return 0.0
    ngrams = [' '.join(words[i:i+window_size]) for i in range(len(words)-window_size+1)]
    if not ngrams:
        return 0.0
    counts = Counter(ngrams)
    total_ngrams = len(ngrams)
    entropy = 0.0
    for count in counts.values():
        p = count / total_ngrams
        entropy -= p * math.log2(p)
    return entropy

def analyze_boredom_correlation(db_path: str):
    """
    Analyzes the correlation between stagnation metrics (Boredom Index)
    and the generation of meta_knowledge.
    """
    if not os.path.exists(db_path):
        print(f"Error: Vector store not found at {db_path}")
        return

    try:
        with open(db_path, "r", encoding="utf-8") as f:
            memory = json.load(f)
    except json.JSONDecodeError:
        print("Error: Could not decode vector store JSON.")
        return

    meta_knowledges = [m for m in memory if m.get("metadata", {}).get("type") == "meta_knowledge"]
    internal_monologues = [m for m in memory if m.get("metadata", {}).get("type") == "internal_monologue"]

    print("=== Correlation Analysis: Entropy & Meta-Knowledge ===")
    print(f"Total entries in vector store: {len(memory)}")
    print(f"Internal Monologues found: {len(internal_monologues)}")
    print(f"Meta-Knowledge rules found: {len(meta_knowledges)}")

    if not internal_monologues and not meta_knowledges:
        print("Not enough data to calculate correlations. Please generate more internal monologues and run sleep cycles.")
        return

    monologue_entropies = [calculate_shannon_entropy(m.get("text", "")) for m in internal_monologues]
    meta_entropies = [calculate_shannon_entropy(m.get("text", "")) for m in meta_knowledges]

    avg_monologue_entropy = sum(monologue_entropies) / len(monologue_entropies) if monologue_entropies else 0.0
    avg_meta_entropy = sum(meta_entropies) / len(meta_entropies) if meta_entropies else 0.0

    print(f"\nAverage Entropy (Internal Monologue): {avg_monologue_entropy:.4f}")
    print(f"Average Entropy (Meta-Knowledge): {avg_meta_entropy:.4f}")

    # Calculate stagnation reduction
    if avg_monologue_entropy > 0 and avg_meta_entropy > 0:
        entropy_ratio = avg_meta_entropy / avg_monologue_entropy
        print(f"Entropy Ratio (Meta/Monologue): {entropy_ratio:.2f}")

        if entropy_ratio > 1.0:
            print("\nConclusion: Meta-knowledge successfully condenses information and INCREASES semantic diversity, overcoming stagnation (boredom).")
        else:
            print("\nConclusion: Meta-knowledge maintains or decreases semantic diversity. The system might be over-compressing or repeating itself.")

if __name__ == "__main__":
    # Pointing to the actual vector store path
    db_path = os.path.join(os.path.dirname(__file__), "..", "data", "vector_memory.json")
    analyze_boredom_correlation(db_path)
