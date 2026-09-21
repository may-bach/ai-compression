"""
Baseline LSAQ Scorer — unweighted Jaccard similarity between layer input/output token sets.

    J(A_l, B_l) = |A_l ∩ B_l| / |A_l ∪ B_l|

High score → layer barely changes token distribution (low importance → lower precision).
Low score  → layer significantly transforms tokens (high importance → higher precision).
"""

from __future__ import annotations

from typing import Dict, List, Tuple


def jaccard_similarity(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)


def score_layer_lsaq(
    input_token_sets: List[List[str]],
    output_token_sets: List[List[str]],
) -> float:
    """Mean Jaccard similarity across all calibration samples for one layer."""
    assert len(input_token_sets) == len(output_token_sets)
    scores = [
        jaccard_similarity(set(inp), set(out))
        for inp, out in zip(input_token_sets, output_token_sets)
    ]
    return sum(scores) / len(scores) if scores else 0.0


def score_all_layers_lsaq(
    layer_token_sets: Dict[int, Tuple[List[List[str]], List[List[str]]]],
) -> Dict[int, float]:
    """Compute LSAQ scores for all layers. Returns layer_idx → score."""
    return {
        idx: score_layer_lsaq(inp, out)
        for idx, (inp, out) in sorted(layer_token_sets.items())
    }
