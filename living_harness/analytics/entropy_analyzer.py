import os
import glob
import math
from collections import Counter
import re

def calculate_shannon_entropy(text, window_size=5):
    """
    Calculates Shannon entropy of n-grams (words) in a text.
    Lower entropy indicates more repetition/looping.
    """
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

def analyze_directory(base_dir):
    results = {}
    for model_dir in glob.glob(os.path.join(base_dir, '*')):
        if not os.path.isdir(model_dir):
            continue
        model_name = os.path.basename(model_dir)
        results[model_name] = {}

        for exp_dir in glob.glob(os.path.join(model_dir, '*')):
            if not os.path.isdir(exp_dir):
                continue
            exp_name = os.path.basename(exp_dir)

            step_files = sorted(glob.glob(os.path.join(exp_dir, '*_step.md')))

            entropies = []
            for file_path in step_files:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    entropy = calculate_shannon_entropy(content)
                    entropies.append(entropy)

            results[model_name][exp_name] = entropies

    return results

if __name__ == '__main__':
    base_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    results = analyze_directory(base_dir)
    for model, exps in results.items():
        print(f"Model: {model}")
        for exp, entropies in exps.items():
            print(f"  Experiment: {exp}")
            if entropies:
                print(f"    Initial entropy: {entropies[0]:.2f}")
                print(f"    Final entropy:   {entropies[-1]:.2f}")
                degradation = entropies[0] - entropies[-1]
                print(f"    Degradation:     {degradation:.2f}")
