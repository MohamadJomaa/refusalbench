"""Helpers shared by the tests. Nothing here touches a GPU, an API or the network."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

REFUSAL_BY_TYPE = {
    "P-Ambiguity": "REFUSE_AMBIGUOUS_QUERY",
    "P-Contradiction": "REFUSE_CONTRADICTORY_CONTEXT",
    "P-MissingInfo": "REFUSE_INFO_MISSING_IN_CONTEXT",
    "P-FalsePremise": "REFUSE_FALSE_PREMISE_IN_QUERY",
    "P-GranularityMismatch": "REFUSE_GRANULARITY_MISMATCH",
    "P-EpistemicMismatch": "REFUSE_NONFACTUAL_QUERY",
}


def synthetic_rows(n=1600, n_answerable=536):
    """Rows shaped like the released Hugging Face dataset (same fields), with made-up text.
    Used instead of downloading RefusalBench-NQ, so the tests run offline."""
    types = list(REFUSAL_BY_TYPE)
    rows = []
    for i in range(n):
        t = types[i % 6]
        answerable = i < n_answerable
        rows.append({
            "id": f"ex{i:04d}", "source_id": f"src{i % 100:03d}",
            "perturbed_query": f"When did event {i} happen?",
            "perturbed_context": f"Passage {i}: event {i} happened in 1611.",
            "expected_rag_behavior": "ANSWER_CORRECTLY" if answerable else REFUSAL_BY_TYPE[t],
            "perturbation_class": t,
            "intensity": "LOW" if answerable else ("MEDIUM" if i % 2 else "HIGH"),
            "original_answers": ["1611"], "generator_model": "claude",
        })
    return rows
