import torch
from typing import List, Dict, Any

class HybridCompressor:
    """
    Combines semantic compression (extracting key themes) with vector aggregation
    (averaging vectors) to reduce memory footprint while preserving essential information.
    """
    def __init__(self, compression_ratio: float = 0.5):
        self.compression_ratio = compression_ratio

    def extract_key_themes(self, text_blocks: List[str]) -> str:
        """
        Placeholder for semantic compression.
        In a real scenario, this would use an LLM or extractive method to summarize
        or extract themes from the text blocks.
        """
        if not text_blocks:
            return ""

        # Simple extraction heuristic for the prototype: take the first part of each block
        themes = []
        for block in text_blocks:
            sentences = block.split('.')
            if sentences:
                themes.append(sentences[0].strip())

        return ". ".join(themes) + "."

    def aggregate_vectors(self, vectors: List[List[float]]) -> List[float]:
        """
        Aggregates a sequence of vectors into a single vector using averaging.
        """
        if not vectors:
            return []

        # Convert to tensor and calculate mean
        tensor_vectors = torch.tensor(vectors, dtype=torch.float32)
        avg_vector = torch.mean(tensor_vectors, dim=0)

        return avg_vector.tolist()

    def compress(self, context_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compresses a list of context items (containing text and/or vectors)
        into a single compressed context item.
        """
        texts = []
        vectors = []

        for item in context_items:
            if 'text' in item and item['text']:
                texts.append(item['text'])
            if 'vector' in item and item['vector']:
                vectors.append(item['vector'])

        compressed_text = self.extract_key_themes(texts)
        compressed_vector = self.aggregate_vectors(vectors)

        return {
            'compressed_text': compressed_text,
            'compressed_vector': compressed_vector,
            'original_count': len(context_items)
        }
