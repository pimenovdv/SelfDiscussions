import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import json
import os

def run_benchmark():
    model_id = "prithivMLmods/SmolLM2-Rethink-135M"
    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)

    contexts = [
        "User is inactive. Recent conversation was about thermodynamics.",
        "User is sleeping. Last topic was cognitive biases.",
        "No activity for 2 hours. Memory shows interest in evolutionary biology."
    ]

    results = []
    total_time = 0

    print("Running benchmark...")
    for i, ctx in enumerate(contexts):
        prompt = f"<think>\nAnalyze the context: {ctx}\n</think>\nSystem: Based on the analysis, suggest a new internal goal to explore.\nNew Goal:"
        inputs = tokenizer(prompt, return_tensors="pt")

        start = time.time()
        outputs = model.generate(**inputs, max_new_tokens=100, do_sample=True, temperature=0.6)
        end = time.time()

        gen_time = end - start
        total_time += gen_time

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        results.append({
            "context": ctx,
            "response": response,
            "time_sec": round(gen_time, 2)
        })
        print(f"Context {i+1} done in {gen_time:.2f}s")

    os.makedirs("data", exist_ok=True)
    out_file = "data/self_prompting_logs.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "model": model_id,
            "avg_time_sec": round(total_time / len(contexts), 2),
            "results": results
        }, f, indent=2, ensure_ascii=False)

    print(f"Benchmark completed. Results saved to {out_file}")

if __name__ == "__main__":
    run_benchmark()
