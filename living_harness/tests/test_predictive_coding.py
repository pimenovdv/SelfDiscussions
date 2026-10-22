import pytest
from living_harness.memory.vector_store import VectorStore

def test_predict_next_domain():
    store = VectorStore(":memory:")

    # Simulate queries in specific domains
    store.search([0.1], domain="science")
    store.search([0.2], domain="philosophy")
    store.search([0.3], domain="science")
    store.search([0.4], domain="philosophy")
    store.search([0.5], domain="art")

    # Predict based on 'science'
    assert store.predict_next_domain("science") == "philosophy"

    # Predict based on 'philosophy'
    # It transitioned to 'science' once, and to 'art' once.
    # Let's add another transition to make it deterministic
    store.search([0.6], domain="philosophy")
    store.search([0.7], domain="art")

    assert store.predict_next_domain("philosophy") == "art"

    # Predict for unknown domain
    assert store.predict_next_domain("unknown") is None
