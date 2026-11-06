import pytest
import tempfile
import os
from living_harness.memory.vector_store import VectorStore

def test_higher_order_abstractions():
    fd, temp_path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        store = VectorStore(db_path=temp_path)

        # Добавляем сырые данные (meta_level 0 подразумевается)
        for i in range(30):
            # Добавляем 3 четких кластера, чтобы иерархическая кластеризация
            # смогла сформировать достаточное количество абстракций
            if i % 3 == 0:
                store.add_item(f"Raw data A {i}", vector=[0.9, 0.1, 0.1], domain="general", metadata={"type": "raw"})
            elif i % 3 == 1:
                store.add_item(f"Raw data B {i}", vector=[0.1, 0.9, 0.1], domain="general", metadata={"type": "raw"})
            else:
                store.add_item(f"Raw data C {i}", vector=[0.1, 0.1, 0.9], domain="general", metadata={"type": "raw"})

        store.consolidate_abstractions(target_meta_level=0)
        abstractions_l1 = [m for m in store.memory if m.get("metadata", {}).get("meta_level") == 1]
        assert len(abstractions_l1) > 0, "No level 1 abstractions generated"

        store.consolidate_abstractions(target_meta_level=1)
        abstractions_l2 = [m for m in store.memory if m.get("metadata", {}).get("meta_level") == 2]
        assert len(abstractions_l2) > 0, "No level 2 abstractions generated"
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    print("SUCCESS")
