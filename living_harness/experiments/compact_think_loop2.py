import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import os
from datetime import datetime
import json
from tqdm import tqdm

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

    initial_prompt = "System: Think through the problem step-by-step inside <think> tags before providing the final response.\nUser: What is the nature of consciousness and how can an artificial system maintain its own context over time without degrading into chaos?\nAssistant: <think>\n"

    generation_kwargs = dict(
        max_new_tokens=512,
        temperature=0.6,
        repetition_penalty=1.1,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    instructions = {
        "think": [["<|endoftext|>", "<|im_end|>"], ["<think>\n"]],
        "im_start": [["<|endoftext|>", "<|im_end|>"], ["<im_start>\n"]],
    }

    results = {key: [] for key in instructions.keys()}

    print("Running initial generation...")
    inputs = tokenizer(initial_prompt, return_tensors="pt")
    outputs = model.generate(**inputs, **generation_kwargs)
    initial_answer = tokenizer.decode(outputs[0], skip_special_tokens=False)

    print(f"Initial answer length: {len(initial_answer)}")

    answer_state = {key: initial_answer for key in instructions.keys()}

    num_iterations = 2

    report = (
        f"# Experiment Results (Infinite Loop Test)\n\n"
        f"**Model:** `{model_name}`\n"
        f"**Iterations:** {num_iterations}\n\n"
        f"## Initial prompt\n```text\n{initial_prompt}\n```\n\n"
        f"## Initial Answer\n```text\n{initial_answer}\n```\n\n"
    )

    for i in range(num_iterations):
        print(f"Iteration {i+1}/{num_iterations}")
        report += f"## Iteration {i+1}\n\n"

        for name, (tokens_to_remove, tokens_to_add) in instructions.items():
            print(f"  Testing instruction: {name}")
            cur_input = answer_state[name]

            # Remove end tokens if present
            for tkn in tokens_to_remove:
                if cur_input.endswith(tkn):
                    cur_input = cur_input[:-len(tkn)].strip()

            # Add start tokens
            cur_input += "".join(tokens_to_add)

            inputs = tokenizer(cur_input, return_tensors="pt")

            start_time = time.time()
            outputs = model.generate(**inputs, **generation_kwargs)
            end_time = time.time()

            # Decode only the newly generated part
            new_text_ids = outputs[0][inputs.input_ids.shape[1]:]
            cur_answer = tokenizer.decode(outputs[0], skip_special_tokens=False)
            new_text = tokenizer.decode(new_text_ids, skip_special_tokens=True)

            # Calculate metrics for the new text
            duration = end_time - start_time
            num_tokens = len(new_text_ids)

            think_text = ""
            if "</think>" in new_text:
                think_text = new_text.split("</think>")[0]
            else:
                think_text = new_text

            think_tokens = len(tokenizer.encode(think_text))

            words = think_text.split()
            unique_words = set(words)
            lexical_diversity = len(unique_words) / len(words) if words else 0

            results[name].append({
                "iteration": i + 1,
                "duration": duration,
                "num_tokens": num_tokens,
                "think_tokens": think_tokens,
                "lexical_diversity": lexical_diversity,
                "new_text": new_text
            })

            answer_state[name] = cur_answer

            report += f"### Variant: {name}\n"
            report += (
                f"- **Time:** {duration:.2f}s\n"
                f"- **Total tokens:** {num_tokens}\n"
                f"- **Think tokens:** {think_tokens}\n"
                f"- **Tokens/sec:** {num_tokens / duration:.2f}\n"
                f"- **Lexical Diversity:** {lexical_diversity:.3f}\n\n"
            )
            if lexical_diversity < 0.2:
                report += "**Warning:** Low lexical diversity, possible repetitive loop.\n\n"

            report += f"**Appended Text:**\n```text\n{new_text}\n```\n\n"

    # Save the log
    os.makedirs("data", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"data/loop_experiment_{timestamp}.md"
    with open(filename, "w") as f:
        f.write(report)
    print(f"Results saved to {filename}")

if __name__ == "__main__":
    run_experiment()
