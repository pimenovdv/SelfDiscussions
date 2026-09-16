import pytest
from living_harness.core.decay_mechanisms import RelevanceDecay
import math

def test_relevance_decay_penalty():
    decay = RelevanceDecay(base_decay_rate=0.1)

    # Свежая память (age=0)
    assert decay.calculate_penalty(0) == 1.0

    # Старая память
    penalty_10 = decay.calculate_penalty(10)
    expected = math.exp(-0.1 * 10)
    assert math.isclose(penalty_10, expected)

    # Очень старая память - релевантность стремится к 0
    penalty_100 = decay.calculate_penalty(100)
    assert penalty_100 < 0.01

def test_relevance_decay_noise():
    decay = RelevanceDecay(noise_variance_factor=0.1)
    vector = [1.0, 1.0]

    # Свежая память - шума нет
    noisy_0 = decay.apply_thermal_noise(vector, 0)
    assert noisy_0 == vector

    # Старая память - добавляется шум
    noisy_10 = decay.apply_thermal_noise(vector, 10)
    assert len(noisy_10) == 2
    assert noisy_10 != vector # С очень большой вероятностью значения изменятся
