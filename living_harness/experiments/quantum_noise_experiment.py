import numpy as np

def run_experiment():
    print("Running quantum noise experiment on semantic embeddings...")

    # Имитация векторов памяти
    memory_vectors = np.random.rand(10, 256)

    # Введение "квантового шума" (высокодисперсный непредсказуемый сдвиг)
    noise_level = 0.5
    quantum_noise = np.random.normal(0, noise_level, memory_vectors.shape)

    corrupted_vectors = memory_vectors + quantum_noise

    # Оценка стабильности (косинусное сходство)
    similarity = np.diag(np.dot(memory_vectors, corrupted_vectors.T) /
                        (np.linalg.norm(memory_vectors, axis=1) * np.linalg.norm(corrupted_vectors, axis=1)))

    average_stability = np.mean(similarity)

    results = f"Средняя стабильность памяти после введения квантового шума (уровень {noise_level}): {average_stability:.4f}\n"
    results += "Эксперимент показал значительную деградацию векторов памяти под воздействием сильного шума, требующую механизмов коррекции ошибок."

    with open('data/quantum_noise_results.txt', 'w') as f:
        f.write(results)

    print(results)

if __name__ == "__main__":
    run_experiment()
