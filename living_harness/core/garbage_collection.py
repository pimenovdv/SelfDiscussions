import time
import threading
from typing import List, Dict, Any
from living_harness.memory.vector_store import VectorStore
from living_harness.memory.partitioning import PartitionManager
from living_harness.core.semantic_autoencoder import SemanticAutoencoderECC

class ColdStorageGC:
    def __init__(self, vector_store: VectorStore, partition_manager: PartitionManager):
        self.vector_store = vector_store
        self.partition_manager = partition_manager
        self._stop_event = threading.Event()
        self._thread = None
        self.autoencoder = SemanticAutoencoderECC()

    @property
    def is_running(self):
        return not self._stop_event.is_set() and self._thread is not None and self._thread.is_alive()

    @is_running.setter
    def is_running(self, value):
        if value:
            self._stop_event.clear()
        else:
            self._stop_event.set()

    def compress_and_archive(self, cold_memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        defragmented = []
        for mem in cold_memories:
            if "vector" in mem and isinstance(mem["vector"], list):
                try:
                    corrected_vector = self.autoencoder.detect_and_correct(mem["vector"])
                    mem["vector"] = corrected_vector
                except Exception as e:
                    pass

                is_duplicate = False
                for existing_mem in defragmented:
                    if "vector" in existing_mem:
                        score = self.vector_store._cosine_similarity(mem["vector"], existing_mem["vector"])
                        if score > 0.95:
                            is_duplicate = True
                            break
                if not is_duplicate:
                    defragmented.append(mem)
            else:
                defragmented.append(mem)
        return defragmented

    def gc_cycle(self):
        # Gather cold memories
        current_time = time.time()
        hot_window = self.partition_manager.hot_window_seconds

        hot_memories = []
        cold_memories = []

        for m in self.vector_store.memory:
            timestamp = m.get("metadata", {}).get("timestamp", current_time)
            if (current_time - timestamp) <= hot_window:
                hot_memories.append(m)
            else:
                cold_memories.append(m)

        if len(cold_memories) > 0:
            compressed_cold = self.compress_and_archive(cold_memories)
            self.vector_store.memory = hot_memories + compressed_cold
            self.vector_store._save()

    def start_background_gc(self, interval_seconds: float = 3600):
        self._stop_event.clear()

        def run_loop():
            while not self._stop_event.is_set():
                self.gc_cycle()
                self._stop_event.wait(interval_seconds)

        self._thread = threading.Thread(target=run_loop, daemon=True)
        self._thread.start()

    def stop_background_gc(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join()
