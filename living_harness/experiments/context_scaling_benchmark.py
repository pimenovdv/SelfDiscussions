import time
import sys
import os
import datetime

# Добавляем корневой каталог в sys.path для корректного импорта living_harness.*
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from living_harness.memory.context_manager import ContextManager
from transformers import pipeline

def run_benchmark():
    print("Запуск бенчмарка: Масштабирование памяти с увяданием")

    # Инициализация
    manager = ContextManager(system_prompt="Тестовый системный промпт", max_window_tokens=100000)

    # Добавление большого количества воспоминаний (имитация долгой работы)
    num_memories = 10000
    print(f"Генерация {num_memories} воспоминаний...")
    start_time = time.time()

    current_time = datetime.datetime.now()
    # Заполняем память с метками времени от "несколько часов назад" до "сейчас"
    for i in range(num_memories):
        # Размазываем по времени, от -120 минут до 0
        minutes_ago = 120 * (1 - (i / num_memories))
        timestamp = (current_time - datetime.timedelta(minutes=minutes_ago)).isoformat()

        # Обходим встроенный add_memory, чтобы установить произвольный timestamp
        manager.memory.append({
            "timestamp": timestamp,
            "content": f"Воспоминание {i}: это тестовая запись для оценки масштабирования."
        })

    gen_time = time.time() - start_time
    print(f"Генерация завершена за {gen_time:.2f} сек.")

    print(f"Всего записей до увядания: {len(manager.memory)}")

    # Бенчмарк формирования промпта (включает логику увядания)
    print("Применение увядания и формирование промпта...")
    start_time = time.time()
    prompt = manager.build_prompt()
    build_time = time.time() - start_time

    print(f"Промпт сформирован за {build_time:.4f} сек.")
    print(f"Осталось записей после увядания: {len(manager.memory)}")
    print(f"Размер промпта (символов): {len(prompt)}")

    print("Запуск инференса на компактной модели...")
    try:
        # Используем ультракомпактную модель из списка: prithivMLmods/SmolLM2-Rethink-135M
        pipe = pipeline("text-generation", model="prithivMLmods/SmolLM2-Rethink-135M", max_new_tokens=50)
        # Отправляем только конец промпта, чтобы не перегружать 135M модель
        inference_start = time.time()
        output = pipe(prompt[-1000:], num_return_sequences=1)
        inference_time = time.time() - inference_start
        print(f"Инференс завершен за {inference_time:.2f} сек.")
    except Exception as e:
        print(f"Ошибка инференса: {e}")
        inference_time = 0.0

    # Сохраняем результаты
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    results_path = os.path.join(data_dir, "context_scaling_results.txt")
    with open(results_path, "w", encoding="utf-8") as f:
        f.write("=== Результаты бенчмарка масштабирования памяти ===\n")
        f.write(f"Исходное количество записей: {num_memories}\n")
        f.write(f"Время генерации записей: {gen_time:.2f} сек.\n")
        f.write(f"Время применения увядания и сборки: {build_time:.4f} сек.\n")
        f.write(f"Осталось записей после фильтрации: {len(manager.memory)}\n")
        f.write(f"Общий размер итогового контекста: {len(prompt)} символов.\n")
        f.write(f"Время инференса на компактной модели: {inference_time:.2f} сек.\n")

if __name__ == "__main__":
    run_benchmark()
