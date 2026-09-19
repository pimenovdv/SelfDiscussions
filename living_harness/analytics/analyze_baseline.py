import os
import json

def main():
    results_dir = "living_harness/results"
    stats = {}
    for root, dirs, files in os.walk(results_dir):
        md_files = [f for f in files if f.endswith('.md')]
        if not md_files:
            continue

        parts = root.split(os.sep)
        if len(parts) >= 3:
            key = f"{parts[-2]}/{parts[-1]}"
            if key not in stats:
                stats[key] = {'steps': 0, 'estimated_tokens': 0, 'avg_tokens_per_step': 0}

            total_toks = 0
            for md in md_files:
                filepath = os.path.join(root, md)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        text = f.read()
                        # simple estimation: 4 chars per token roughly
                        total_toks += len(text) // 4
                except Exception:
                    pass
                stats[key]['steps'] += 1

            stats[key]['estimated_tokens'] += total_toks
            if stats[key]['steps'] > 0:
                stats[key]['avg_tokens_per_step'] = stats[key]['estimated_tokens'] / stats[key]['steps']

    report_path = "living_harness/data/baseline_analysis.json"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding='utf-8') as f:
        json.dump(stats, f, indent=4, ensure_ascii=False)

    print(f"Baseline analysis saved to {report_path}")

if __name__ == "__main__":
    main()
