import os
import sys
import json
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from living_harness.analytics.autonomy_metrics import calculate_reasoning_density

def run_benchmark():
    model_name = "prithivMLmods/SmolLM2-Rethink-135M"
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, device_map="cpu")

    messages = [
        {"role": "system", "content": "Think through the problem step-by-step inside <think> tags before providing the final response."},
        {"role": "user", "content": "How would an autonomous AI maintain its own context over a 24-hour period?"}
    ]

    try:
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        input_text = "<think>\n"

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    print("Starting continuous reasoning benchmark...")

    logs = []
    start_time = time.time()

    for step in range(3):
        print(f"Iteration {step+1}...")
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.6
        )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)

        if generated_text.endswith("<|im_end|>") or generated_text.endswith("<|endoftext|>"):
            generated_text = generated_text.replace("<|im_end|>", "").replace("<|endoftext|>", "") + "\n<think>\n"

        logs.append({
            "step": step,
            "text": generated_text,
            "timestamp": time.time()
        })

        inputs = tokenizer(generated_text, return_tensors="pt").to(model.device)

    end_time = time.time()

    final_text = logs[-1]["text"]
    density = calculate_reasoning_density(final_text)

    results = {
        "model": model_name,
        "iterations": 3,
        "total_time_sec": end_time - start_time,
        "reasoning_density": density,
        "logs": logs
    }

    # Use path relative to __file__ to save the artifact
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    os.makedirs(data_dir, exist_ok=True)
    output_path = os.path.join(data_dir, "autonomy_benchmark_results.json")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"Benchmark completed. Results saved to {output_path}")
    print(f"Reasoning density: {density:.2f}")

if __name__ == "__main__":
    run_benchmark()
