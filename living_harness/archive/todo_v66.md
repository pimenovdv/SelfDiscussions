# План развития и исследований (TODO) - Этап 66: Регуляция абстракций (Abstraction Pruning)

- [x] 1. **Инженерия: Очистка устаревших концептов**
  - Реализовать метод `prune_stale_abstractions` в `VectorStore` (Код: `living_harness/memory/vector_store.py`), который удаляет абстрактные узлы (`is_abstract=True`), чей `importance_weight` упал ниже порога. (Артефакт: `living_harness/memory/vector_store.py`)
- [x] 2. **Эксперимент: Проверка забывания абстракций**
  - Написать тест `test_abstraction_pruning.py`, который симулирует деградацию весов для абстрактных концептов и проверяет их успешное удаление. (Артефакт: `living_harness/tests/test_abstraction_pruning.py`)
- [x] 3. **Исследование: Динамика забывания концептов**
  - Написать драфт статьи `concept_forgetting_dynamics.md`, объясняющий важность забывания неактуальных абстракций для поддержания чистоты "горячего" контекста ИИ. (Артефакт: `living_harness/articles/concept_forgetting_dynamics.md`)
