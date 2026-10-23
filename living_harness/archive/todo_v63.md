# План развития и исследований (TODO) - Этап 63: Анализ графов доменов (Domain Graph Analysis)

- [x] 1. **Инженерия: Граф переходов**
  - Реализовать метод `export_domain_graph` в `VectorStore` (Код: `living_harness/memory/vector_store.py`), который экспортирует матрицу переходов между доменами в формате графа (например, NetworkX) для последующего анализа. (Артефакт: `living_harness/memory/vector_store.py`)
- [x] 2. **Эксперимент: Центральность доменов**
  - Написать скрипт анализа графа, чтобы вычислить PageRank или Betweenness Centrality для доменов, определяя ключевые "хабы" знаний. (Код: `living_harness/experiments/test_domain_centrality.py`) (Артефакт: `living_harness/experiments/test_domain_centrality.py`)
- [x] 3. **Исследование: Топология памяти ИИ**
  - Написать драфт статьи "Топология памяти ИИ: Анализ графовых структур переходов между доменами знаний". (Код: `living_harness/articles/domain_topology_draft.md`) (Артефакт: `living_harness/articles/domain_topology_draft.md`)
