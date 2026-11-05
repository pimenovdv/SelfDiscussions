from typing import List, Dict, Callable
import time
import math

def calculate_ndcg(retrieved_ids: List[str], expected_ids: List[str], k: int) -> float:
    """Calculates Normalized Discounted Cumulative Gain (NDCG) at k."""
    dcg = 0.0
    for i in range(min(k, len(retrieved_ids))):
        if retrieved_ids[i] in expected_ids:
            # Relevance is binary (1 if present, 0 otherwise)
            dcg += 1.0 / math.log2(i + 2)

    idcg = 0.0
    for i in range(min(k, len(expected_ids))):
        idcg += 1.0 / math.log2(i + 2)

    if idcg == 0:
        return 0.0
    return dcg / idcg

def evaluate_semantic_retrieval(vector_store, queries: List[List[float]], expected_ids: List[List[str]], top_k: int = 5) -> Dict[str, float]:
    """
    Оценивает скорость и качество поиска (NDCG@k, Recall@k) по векторной базе.
    """
    start_time = time.time()
    recall_sum = 0.0
    ndcg_sum = 0.0

    for query, expected in zip(queries, expected_ids):
        results = vector_store.search(query, top_k=top_k)
        retrieved_ids = [r.get("id") for r in results if r.get("id") is not None]

        # Calculate Recall
        hits = sum(1 for e in expected if e in retrieved_ids)
        recall_sum += hits / len(expected) if expected else 0

        # Calculate NDCG
        ndcg_sum += calculate_ndcg(retrieved_ids, expected, top_k)

    end_time = time.time()

    avg_recall = recall_sum / len(queries) if queries else 0.0
    avg_ndcg = ndcg_sum / len(queries) if queries else 0.0
    latency = (end_time - start_time) / len(queries) if queries else 0.0

    return {
        "ndcg_at_k": avg_ndcg,
        "recall_at_k": avg_recall,
        "latency_sec_per_query": latency
    }
