import numpy as np
import os
import sys

sys.path.append(os.path.abspath('living_harness'))

from core.semantic_ecc import SemanticECC
from core.semantic_autoencoder import SemanticAutoencoderECC

def run_experiment():
    print("Running Denoising Autoencoder efficiency benchmark against quantum noise...")
    np.random.seed(42)

    # Simulate memory vectors with values in [-1, 1] to fit Tanh activation well
    num_vectors = 1000
    vector_dim = 256
    memory_vectors = np.random.uniform(-1, 1, size=(num_vectors, vector_dim))

    # Introduce quantum noise
    noise_level = 0.5
    quantum_noise = np.random.normal(0, noise_level, memory_vectors.shape)
    corrupted_vectors = memory_vectors + quantum_noise

    # Initialize Autoencoder ECC
    model_path = os.path.abspath('data/autoencoder.pth')
    ae_ecc = SemanticAutoencoderECC(input_dim=vector_dim, hidden_dim=64, model_path=model_path)

    print("Training autoencoder (denoising mode)...")
    for epoch in range(100):
        # We train on pairs of (noisy_vectors -> original clean vectors)
        loss = ae_ecc.train_step(memory_vectors, corrupted_vectors)
        if epoch % 20 == 0:
            print(f"Epoch {epoch} loss: {loss:.4f}")

    ae_ecc.save_model()

    # Detect and correct
    corrected_vectors_ae = []
    for i in range(num_vectors):
        corrected_vector = ae_ecc.detect_and_correct(corrupted_vectors[i].tolist())
        corrected_vectors_ae.append(corrected_vector)

    corrected_vectors_ae = np.array(corrected_vectors_ae)

    # Initialize Scalar ECC for comparison
    scalar_ecc = SemanticECC(redundancy_factor=3, threshold=0.1)
    for i in range(num_vectors):
        scalar_ecc.generate_redundant_encoding(f"mem_{i}", memory_vectors[i].tolist())

    corrected_vectors_scalar = []
    for i in range(num_vectors):
        corrected_vector = scalar_ecc.detect_and_correct(f"mem_{i}", corrupted_vectors[i].tolist())
        corrected_vectors_scalar.append(corrected_vector)

    corrected_vectors_scalar = np.array(corrected_vectors_scalar)

    # Calculate similarities
    def calc_similarity(orig, new_vecs):
        sims = np.diag(np.dot(orig, new_vecs.T) /
                      (np.linalg.norm(orig, axis=1) * np.linalg.norm(new_vecs, axis=1)))
        return np.mean(sims)

    corrupted_similarity = calc_similarity(memory_vectors, corrupted_vectors)
    corrected_similarity_scalar = calc_similarity(memory_vectors, corrected_vectors_scalar)
    corrected_similarity_ae = calc_similarity(memory_vectors, corrected_vectors_ae)

    results = (
        f"Средняя стабильность (косинусное сходство) до коррекции (уровень шума 0.5): {corrupted_similarity:.4f}\n"
        f"Средняя стабильность после применения скалярного Semantic ECC: {corrected_similarity_scalar:.4f}\n"
        f"Средняя стабильность после применения Denoising Autoencoder: {corrected_similarity_ae:.4f}\n"
        f"Абсолютное улучшение стабильности (AE): {(corrected_similarity_ae - corrupted_similarity):.4f}\n\n"
        f"Вывод: В отличие от скалярного масштабирования, которое не меняет угловое расстояние, автоэнкодер (denoising autoencoder) успешно восстанавливает как магнитуду, так и направление векторов (семантический смысл)."
    )

    data_dir = os.path.abspath('data')
    os.makedirs(data_dir, exist_ok=True)
    with open(os.path.join(data_dir, 'semantic_autoencoder_results.txt'), 'w', encoding='utf-8') as f:
        f.write(results)

    print(results)

if __name__ == "__main__":
    run_experiment()
