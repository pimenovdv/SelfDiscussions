import unittest
import numpy as np
import os
from living_harness.core.garbage_collection import ColdStorageGC
from living_harness.memory.vector_store import VectorStore
from living_harness.memory.partitioning import PartitionManager

class TestSemanticDegradation(unittest.TestCase):
    def test_degradation_at_high_capacity(self):
        store_path = "living_harness/data/test_vector_memory_deg.json"
        store = VectorStore(store_path)
        partition_manager = PartitionManager()
        gc = ColdStorageGC(store, partition_manager)

        # Generate 100 similar vectors with dimension 256
        np.random.seed(42)
        base_vector = np.array([0.5] * 256)
        memories = []
        for i in range(100):
            noise = np.random.normal(0, 0.15, 256) # Increased noise to cross threshold boundary
            vec = (base_vector + noise).tolist()
            memories.append({"text": f"memory {i}", "vector": vec, "metadata": {"timestamp": 0}})

        compressed_small = gc.compress_and_archive(memories.copy(), current_storage_size=0)
        compressed_large = gc.compress_and_archive(memories.copy(), current_storage_size=10000)

        self.assertLess(len(compressed_large), len(compressed_small))

        if os.path.exists(store_path):
            os.remove(store_path)

if __name__ == "__main__":
    unittest.main()
