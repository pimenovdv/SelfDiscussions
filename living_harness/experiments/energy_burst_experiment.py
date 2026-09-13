import time
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.burst_inference import BurstInferenceManager

def simulate_generation(tokens: int, continuous: bool):
    """
    Симулирует задержку генерации для оценки времени.
    В непрерывном режиме генерируются все токены.
    В прерывистом - генерация быстрее за счет отброшенного 'шума'.
    """
    delay_per_token = 0.05
    time.sleep(tokens * delay_per_token if continuous else tokens * delay_per_token * 0.2)

def run_experiment():
    print("Запуск эксперимента: Непрерывная vs Прерывистая генерация")
    manager = BurstInferenceManager(surprise_threshold=0.6)

    events = [
        "Привет, как дела?", # Высокий сюрприз
        "Нормально",         # Низкий сюрприз (слова пересекаются мало, но добавим общий контекст ниже)
        "Я тут подумал...",  # Средний сюрприз
        "Срочная новость: открыт новый закон физики!" # Высокий сюрприз
    ]

    # 1. Непрерывный режим
    print("\n--- Непрерывный режим ---")
    start_time = time.time()
    continuous_tokens_generated = 0
    context_cont = ""
    for event in events:
        print(f"Обработка: {event}")
        # Модель постоянно генерирует <think>
        simulate_generation(50, continuous=True)
        continuous_tokens_generated += 50
        context_cont += event + " "

    continuous_time = time.time() - start_time
    print(f"Время работы (непрерывный): {continuous_time:.2f} сек")
    print(f"Сгенерировано токенов 'шума': {continuous_tokens_generated}")

    # 2. Прерывистый режим (Burst)
    print("\n--- Прерывистый режим (Burst) ---")
    start_time = time.time()
    burst_tokens_generated = 0
    context_burst = "Привет, как дела? Нормально Я тут подумал..." # Предустановленный контекст для снижения новизны

    for event in events:
        print(f"Обработка: {event}")
        if manager.should_trigger_burst(context_burst, event):
            print(f" > Триггер сработал! Запуск burst-размышлений.")
            manager.trigger_burst(event)
            simulate_generation(50, continuous=True) # Полноценная генерация при триггере
            burst_tokens_generated += 50
        else:
            print(f" > Триггер пропущен. Экономия энергии.")
            simulate_generation(10, continuous=False) # Быстрый пропуск

        context_burst += " " + event

    burst_time = time.time() - start_time
    print(f"Время работы (прерывистый): {burst_time:.2f} сек")
    print(f"Сгенерировано токенов по делу: {burst_tokens_generated}")

    print("\n--- Результаты ---")
    time_saved = continuous_time - burst_time
    print(f"Сэкономлено времени: {time_saved:.2f} сек ({(time_saved/continuous_time)*100:.1f}%)")

if __name__ == "__main__":
    run_experiment()
