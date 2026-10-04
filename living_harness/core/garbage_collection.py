import time
import threading
from typing import List, Dict, Any
from living_harness.memory.vector_store import VectorStore
from living_harness.memory.partitioning import PartitionManager

class ColdStorageGC:
    def __init__(self, vector_store: VectorStore, partition_manager: PartitionManager):
        self.vector_store = vector_store
        self.partition_manager = partition_manager
        self._stop_event = threading.Event()
        self._thread = None

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
        # Dummy compression for now: just sort by timestamp or keep as is.
        # Ideally, we would compress semantic duplicates.
        # For our GC, let's say we remove items that are extremely old and not accessed (if we tracked access).
        # We'll just return a 'defragmented' list. In a real scenario, this merges vectors.
        defragmented = []
        for mem in cold_memories:
            # Fake deduplication logic
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
