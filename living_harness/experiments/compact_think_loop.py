import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import os
from datetime import datetime

def run_experiment():
    model_name = "prithivMLmods/SmolLM2-Rethink-360M"
    print(f"Loading model {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="cpu",
        torch_dtype=torch.float32,
    )
    print("Model loaded.")

    prompt = "System: Think through the problem step-by-step inside <think> tags before providing the final response.\nUser: What is the nature of consciousness and how can an artificial system maintain its own context over time without degrading into chaos?\nAssistant: <think>\n"
    inputs = tokenizer(prompt, return_tensors="pt")

    start_time = time.time()
    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        temperature=0.6,
        repetition_penalty=1.1,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    end_time = time.time()

    generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    duration = end_time - start_time
    num_tokens = len(outputs[0]) - inputs.input_ids.shape[1]

    think_text = ""
    if "</think>" in generated_text:
        think_text = generated_text.split("</think>")[0]
    else:
        think_text = generated_text

    think_tokens = len(tokenizer.encode(think_text))

    words = think_text.split()
    unique_words = set(words)
    lexical_diversity = len(unique_words) / len(words) if words else 0

    report = (
        "# Experiment Results\n\n"
        f"**Model:** `{model_name}`\n\n"
        f"- **Total generation time:** {duration:.2f} seconds\n"
        f"- **Total generated tokens:** {num_tokens}\n"
        f"- **Think process tokens:** {think_tokens}\n"
        f"- **Tokens per second:** {num_tokens / duration:.2f}\n"
        f"- **Lexical Diversity:** {lexical_diversity:.3f}\n\n"
    )
    if lexical_diversity < 0.2:
        report += "**Warning:** Low lexical diversity, possible repetitive loop.\n\n"

    report += "## Full Output\n\n```text\n"
    report += generated_text + "\n```\n"

    print(report)

    # Save the log
    os.makedirs("data", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"data/experiment_{timestamp}.md"
    with open(filename, "w") as f:
        f.write(report)
    print(f"Results saved to {filename}")

if __name__ == "__main__":
    run_experiment()
