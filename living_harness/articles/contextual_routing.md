# Contextual Routing in Artificial Cognitive Architecture

**Denis, Lead Data Science & Research Engineer**
**"Living AI Harness" Project**

## Abstract

As the knowledge graph and vector memory of a "living" artificial intelligence grow, a critical bottleneck emerges: searching and comparing concepts across the entirety of memory space becomes computationally expensive and cognitively noisy. In biological brains, specific regions or circuits are activated based on context, allowing for highly efficient, targeted processing. We propose and implement a mechanism termed *Contextual Routing* to simulate this biological efficiency within a continuous-learning AI architecture.

## The Problem of Global Search

Standard vector database implementations rely on calculating the cosine similarity between a query vector and all (or a large subset of) vectors in the store. As an agent explores diverse domains—such as `physics`, `philosophy`, and `personal_metrics`—performing a global search for a domain-specific query introduces two primary issues:
1. **Computational Inefficiency:** Exhaustive distance calculations scale linearly with memory size, slowing down the background curiosity cycle and reactive responses.
2. **Cognitive Noise:** Unrelated domains may contain vectors that coincidentally align mathematically but lack semantic relevance to the current cognitive task, leading to "hallucinated" connections or diluted context windows.

## Mechanism: Contextual Routing

To address these challenges, we implemented the `contextual_routing` method. Before executing a granular search or triggering spreading activation, the system evaluates the incoming `query_vector` against aggregated representations of known domains.

### Algorithm

1. **Domain Aggregation:** The vectors of all memory items are grouped by their `domain` label. A mean vector (centroid) $\mu_d$ is calculated for each domain $d$.
2. **Relevance Scoring:** The cosine similarity between the query vector $Q$ and each domain centroid $\mu_d$ is calculated.
3. **Thresholding:** If the similarity exceeds a predefined `similarity_threshold` (e.g., 0.5), the domain is marked as "relevant".
4. **Routing:** The search space is restricted exclusively to memory items belonging to the relevant domains. If no domain crosses the threshold, the system defaults to a global search to prevent complete memory failure during highly novel queries.

## Experimental Results

Testing the mechanism (`test_contextual_routing.py`) demonstrated precise isolation of memory subspaces. Given a query highly aligned with a specific domain centroid, the routing mechanism successfully filtered out 100% of items from unrelated domains.

## Conclusion

Contextual Routing represents a significant step towards creating structured, compartmentalized "brain regions" within a unified vector memory. By mimicking the brain's ability to selectively activate relevant neural pathways, we achieve faster retrieval times and cleaner semantic context, enabling more coherent and sustained reasoning in autonomous AI agents.
