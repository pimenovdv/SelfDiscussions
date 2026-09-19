import os
import json
import re
from collections import Counter

def analyze_file(filepath):
    size = os.path.getsize(filepath)
    word_count = 0
    repetitive_ngram_score = 0
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read(50000)
            words = re.findall(r'\b\w+\b', text.lower())
            word_count = len(words)
            if word_count > 10:
                bigrams = list(zip(words, words[1:]))
                most_common = Counter(bigrams).most_common(1)
                if most_common:
                    repetitive_ngram_score = most_common[0][1] / len(bigrams)
    except Exception as e:
        pass

    return size, word_count, repetitive_ngram_score

def main():
    results_dir = "living_harness/results"
    stats = {}

    for root, dirs, files in os.walk(results_dir):
        md_files = [f for f in files if f.endswith('.md')]
        if not md_files:
            continue

        parts = root.split(os.sep)
        if len(parts) >= 3:
            model_name = parts[-2]
            variant = parts[-1]
            key = f"{model_name}/{variant}"

            if key not in stats:
                stats[key] = {'total_steps': 0, 'total_size': 0, 'avg_repetition_score': 0.0}

            total_rep = 0
            for md in md_files:
                filepath = os.path.join(root, md)
                size, wcount, rep_score = analyze_file(filepath)
                stats[key]['total_steps'] += 1
                stats[key]['total_size'] += size
                total_rep += rep_score

            stats[key]['avg_repetition_score'] = total_rep / len(md_files)
            stats[key]['avg_size'] = stats[key]['total_size'] / len(md_files)

    report_path = "living_harness/data/results_analysis.json"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding='utf-8') as f:
        json.dump(stats, f, indent=4, ensure_ascii=False)

    print(f"Analysis saved to {report_path}")

if __name__ == "__main__":
    main()
