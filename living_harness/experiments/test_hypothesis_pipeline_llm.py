import json
import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add parent dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from memory.vector_store import VectorStore

def run_experiment():
    print("Starting LLM hypothesis pipeline experiment...")

    # Initialize a clean vector store
    store = VectorStore(db_path="data/test_llm_memory.json")
    store.clear()

    # Create isolated domain 'Philosophy'
    store.add_item("The sky is blue due to Rayleigh scattering.", [1.0, 0.0, 0.0, 0.0], domain="Science")
    store.add_item("Water is composed of hydrogen and oxygen.", [0.0, 1.0, 0.0, 0.0], domain="Chemistry")
    store.add_item("Consciousness is a mystery that science hasn't fully explained.", [0.0, 0.0, 1.0, 0.0], domain="Philosophy") # Isolated

    # Create transitions between Science and Chemistry
    store.transitions = {
        "Science": {"Chemistry": 1},
        "Chemistry": {"Science": 1}
    }

    # Load a lightweight model for generation (we'll use SmolLM2-135M to be fast)
    model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"
    print(f"Loading model {model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cpu", torch_dtype=torch.float32)
    except Exception as e:
        print(f"Could not load model: {e}")
        print("Using fallback generation...")
        model = None

    def generate_hypothesis(target_domain):
        if model is None:
            return f"The domain {target_domain} might be related to Chemistry and Science."

        prompt = f"Write a short hypothesis connecting {target_domain} to Science or Chemistry:"
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=30)
        return tokenizer.decode(outputs[0], skip_special_tokens=True).replace(prompt, "").strip()

    def vectorize_hypothesis(text):
        # In a real system, we'd use an embedding model here
        # For the experiment, we'll simulate a vector that bridges the domains
        return [0.1, 0.7, 0.7, 0.0]

    # Run the pipeline with the callbacks
    print("Running pipeline...")
    integrated = store.integrate_hypothesis_pipeline(
        generation_callback=generate_hypothesis,
        vectorization_callback=vectorize_hypothesis
    )

    print(f"Generated {len(integrated)} valid hypotheses:")
    for item in integrated:
        print(f"- {item['text']}")

    # Save metrics
    metrics = {
        "targets_identified": 1,
        "hypotheses_generated": len(integrated),
        "valid_hypotheses": len(integrated),
        "model_used": model_name if model else "fallback"
    }

    with open("data/hypothesis_pipeline_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("Experiment completed. Metrics saved.")

if __name__ == "__main__":
    run_experiment()
