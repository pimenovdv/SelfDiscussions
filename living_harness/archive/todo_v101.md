# План развития и исследований (TODO) - Этап 101: R&D, Прототипирование и Бенчмарки

- [x] 1. **Инженерия: Поддержка мультимодальных векторов**
  - Обновить `VectorStore` для поддержки разнородных векторов (изображения, аудио) и их привязки к текстовым концепциям. [vector_store.py](memory/vector_store.py)
- [x] 2. **Эксперимент: Бенчмарк мультимодальной памяти**
  - Создать `test_multimodal_memory.py` для оценки скорости поиска по комбинированным модальностям и сохранить метрики в `data/multimodal_metrics.json`. [test_multimodal_memory.py](experiments/test_multimodal_memory.py)
- [x] 3. **Исследование: Мультимодальная контекстная маршрутизация**
  - Подготовить черновик статьи `articles/multimodal_context_routing.md` о влиянии мультимодальности на эффективность маршрутизации. [multimodal_context_routing.md](articles/multimodal_context_routing.md)
