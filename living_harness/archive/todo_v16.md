# План развития и исследований (TODO) - Этап 16: Тестирование и бенчмаркинг архитектуры Living Harness

- [x] 1. **Разработка системы метрик для оценки автономности**
  - Скрипт `living_harness/analytics/autonomy_metrics.py` создан для расчета idle_time и reasoning_density.
- [x] 2. **Запуск суточного бенчмарка автономной работы** [Артефакт](living_harness/data/autonomy_benchmark_results.json)
  - Провести бенчмарк на одной из моделей из `models.md`.
- [x] 3. **Анализ логов суточного бенчмарка** [Артефакт](living_harness/data/benchmark_analysis_results.json)
  - Выявить проблемы деградации контекста при длительной работе (незакрытые теги, коллапс внимания).
- [x] 4. **Академический драфт: Оценка автономности** [Артефакт](living_harness/articles/autonomy_degradation_draft.md)
  - Подготовлен черновик статьи об автономности в непрерывном цикле генерации и выявленных уязвимостях (attention collapse).
