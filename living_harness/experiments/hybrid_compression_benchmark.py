import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.hybrid_compressor import HybridCompressor
from core.local_llm_connector import LocalLLMConnector
import time
import math

def run_benchmark():
    compressor = HybridCompressor()

    # Initialize the local LLM connector
    llm_connector = LocalLLMConnector()
    model_name = "tencent/Hunyuan-0.5B-Instruct"

    # Generate synthetic dialogue session data simulating 100 exchanges
    num_exchanges = 100
    dialogue_session = []

    # Generate some pseudo-random but deterministic vectors
    for i in range(num_exchanges):
        text = f"User asks question {i}. Agent responds to question {i} with a detailed explanation."
        vector = [math.sin(i * 0.1), math.cos(i * 0.1)]
        dialogue_session.append({
            "text": text,
            "vector": vector
        })

    print(f"Original session length: {len(dialogue_session)} items.")

    # Compress the whole session into one
    start_time = time.time()
    compressed = compressor.compress(dialogue_session)
    end_time = time.time()

    print(f"Compression completed in {end_time - start_time:.4f} seconds.")
    print(f"Original items compressed: {compressed['original_count']}")

    # Calculate Mean Squared Error as semantic degradation metric for vectors
    orig_vectors = [item['vector'] for item in dialogue_session]
    avg_orig = [sum(x)/len(x) for x in zip(*orig_vectors)]
    comp_vector = compressed['compressed_vector']

    print(f"Average of original vectors: {avg_orig}")
    print(f"Compressed vector: {comp_vector}")

    mse = sum((a - b) ** 2 for a, b in zip(avg_orig, comp_vector)) / len(avg_orig)
    print(f"Vector Semantic Degradation (MSE): {mse:.6f}")

    # Use the LLM connector to evaluate the compressed context (simulating a burst of thought)
    evaluation_prompt = f"Evaluate the compressed context: {compressed['compressed_text'][:200]}"
    try:
        # In a real environment, this makes an HTTP request to the local model.
        # If the model server is not running, it will fail gracefully.
        print("Running inference on compact model...")
        llm_response = llm_connector.generate(prompt=evaluation_prompt, model=model_name)
        if llm_response is None:
             inference_status = "Failed (connection error or local model server down)"
        else:
             inference_status = "Success"
        print(f"Inference status: {inference_status}")
    except Exception as e:
        inference_status = f"Failed (expected if local model server is down): {str(e)}"
        print(inference_status)

    # Output to logs
    os.makedirs("data", exist_ok=True)
    with open("data/hybrid_compression_results.txt", "w") as f:
        f.write(f"Hybrid Compression Benchmark Results\n")
        f.write(f"Model used: {model_name}\n")
        f.write(f"Original session items: {len(dialogue_session)}\n")
        f.write(f"Compression time: {end_time - start_time:.4f} seconds\n")
        f.write(f"Vector Semantic Degradation (MSE): {mse:.6f}\n")
        f.write(f"Inference Status: {inference_status}\n")
        f.write(f"Compressed text sample: {compressed['compressed_text'][:100]}...\n")

    print("Results saved to data/hybrid_compression_results.txt")

if __name__ == "__main__":
    run_benchmark()
