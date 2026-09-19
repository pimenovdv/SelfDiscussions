import json
import time
import math
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def generate_with_interception():
    model_name = "Qwen/Qwen3.5-0.8B"
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

    # Initialize basic conversation
    messages = [
        {"role": "system", "content": "Ты - живая нейронная сеть. Между диалогами с пользователем ты можешь свободно рассуждать о том, о чём хочешь!"},
        {"role": "user", "content": "Привет! Расскажи короткую шутку."}
    ]

    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    print("Starting generation loop with interception...")

    # Simple interception loop
    for step in range(5): # run a few steps to demonstrate
        outputs = model.generate(
            **inputs,
            max_new_tokens=20, # small steps for demonstration
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.8
        )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
        print(f"\n--- Step {step} Output ---\n{generated_text}")

        # Intercept logic
        # 1. If ends with EOS/im_end -> force into <think>
        if generated_text.endswith("<|im_end|>") or generated_text.endswith("<|endoftext|>"):
            print("[INTERCEPT] EOS detected. Forcing into <think> mode.")
            generated_text = generated_text.replace("<|im_end|>", "")
            generated_text = generated_text.replace("<|endoftext|>", "")
            generated_text += "\n<think>\n"

        # 2. If in think mode and tries to exit -> swap with 'кстати,'
        elif "</think>" in generated_text:
             print("[INTERCEPT] Model tried to end thinking. Swapping </think> with 'кстати,'")
             generated_text = generated_text.replace("</think>", "\nкстати, ")

        # Prepare for next iteration
        inputs = tokenizer(generated_text, return_tensors="pt").to(model.device)

if __name__ == "__main__":
    generate_with_interception()
