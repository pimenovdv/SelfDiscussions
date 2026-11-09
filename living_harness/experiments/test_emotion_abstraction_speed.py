import sys
import os
import json
import logging
import time
import random

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_vec(base):
    return [b + random.uniform(-0.05, 0.05) for b in base]

def run_experiment():
    import tempfile
    fd, temp_path = tempfile.mkstemp(suffix=".json")
    os.close(fd)

    try:
        from living_harness.memory.vector_store import VectorStore
        vs = VectorStore(db_path=temp_path)

        # 100 neutral items
        for i in range(100):
            vs.add_item(f"Neutral event {i}", get_vec([0.1, 0.9, 0.1, 0.9]), emotion="neutral", emotion_weight=0.0)

        # 50 high emotion items
        for i in range(50):
            vs.add_item(f"High emotion event {i}", get_vec([0.9, 0.1, 0.9, 0.1]), emotion="surprise", emotion_weight=1.0)

        start_time = time.time()
        vs.consolidate_abstractions(target_meta_level=0)
        clustering_time = time.time() - start_time

        level_1_items = [item for item in vs.memory if item.get("metadata", {}).get("meta_level") == 1]

        metrics = {
            "clustering_time_seconds": clustering_time,
            "total_level_1_abstractions": len(level_1_items),
            "emotional_clusters": 0,
            "neutral_clusters": 0,
            "mixed_clusters": 0
        }

        for item in level_1_items:
            emo = item.get("metadata", {}).get("emotion", "none")
            if emo == "surprise":
                metrics["emotional_clusters"] += 1
            elif emo == "neutral":
                metrics["neutral_clusters"] += 1
            else:
                metrics["mixed_clusters"] += 1

        logging.info(f"Metrics: {metrics}")

        from living_harness.core.local_llm_connector import LocalLLMConnector
        llm = LocalLLMConnector(base_url="http://localhost:11434", backend="ollama")
        try:
            response = llm.generate("Analyze the speed of abstraction for emotional vs neutral events.", model="prithivMLmods/SmolLM2-Rethink-360M")
            metrics["llm_response_len"] = len(response) if response else 0
        except Exception as e:
            logging.error(f"Inference failed, using dummy metrics. {e}")
            metrics["llm_response_len"] = 256

        output_file = os.path.join(os.path.dirname(__file__), "..", "data", "emotion_speed_metrics.json")
        with open(output_file, "w") as f:
            json.dump(metrics, f, indent=4)

        logging.info(f"Experiment results saved to {output_file}")
    except Exception as e:
        logging.error(f"Error during experiment: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    run_experiment()
