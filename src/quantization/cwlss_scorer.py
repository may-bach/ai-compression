"""
CW-LSS Scorer — Clinically-Weighted Layer Sensitivity Score (proposed method).

    CW-LSS(l) = Σ_{t ∈ A_l ∩ B_l} w(t)  /  Σ_{t ∈ A_l ∪ B_l} w(t)

where w(t) = alpha for clinically salient tokens (drugs, dosages, lab values,
negation cues) and w(t) = 1.0 otherwise. Layers that corrupt clinical tokens
receive a lower score than their unweighted Jaccard, correctly flagging them
as high-importance and assigning higher quantization precision.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from src.data.clinical_lexicon import get_token_weight


def weighted_jaccard(set_a: set, set_b: set, alpha: float = 5.0) -> float:
    """Clinically-weighted Jaccard similarity between two token sets."""
    if not set_a and not set_b:
        return 1.0
    intersection = set_a & set_b
    union = set_a | set_b
    numerator = sum(get_token_weight(t, alpha) for t in intersection)
    denominator = sum(get_token_weight(t, alpha) for t in union)
    return numerator / denominator if denominator > 0 else 0.0


def score_layer_cwlss(
    input_token_sets: List[List[str]],
    output_token_sets: List[List[str]],
    alpha: float = 5.0,
) -> float:
    """Mean CW-LSS score across all calibration samples for one layer."""
    assert len(input_token_sets) == len(output_token_sets)
    scores = [
        weighted_jaccard(set(inp), set(out), alpha)
        for inp, out in zip(input_token_sets, output_token_sets)
    ]
    return sum(scores) / len(scores) if scores else 0.0


def score_all_layers_cwlss(
    layer_token_sets: Dict[int, Tuple[List[List[str]], List[List[str]]]],
    alpha: float = 5.0,
) -> Dict[int, float]:
    """Compute CW-LSS scores for all layers. Returns layer_idx → score."""
    return {
        idx: score_layer_cwlss(inp, out, alpha)
        for idx, (inp, out) in sorted(layer_token_sets.items())
    }


def compare_scores(
    lsaq_scores: Dict[int, float],
    cwlss_scores: Dict[int, float],
) -> Dict[int, Dict[str, float]]:
    """Return layer_idx → {lsaq, cwlss, delta} where delta = lsaq - cwlss."""
    return {
        idx: {
            "lsaq": round(lsaq_scores[idx], 4),
            "cwlss": round(cwlss_scores.get(idx, lsaq_scores[idx]), 4),
            "delta": round(lsaq_scores[idx] - cwlss_scores.get(idx, lsaq_scores[idx]), 4),
        }
        for idx in sorted(lsaq_scores)
    }
