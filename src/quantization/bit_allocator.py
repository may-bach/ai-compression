"""
Dynamic bit allocator — assigns mixed-precision bit-widths to layers.

Strategy: sort layers by sensitivity score (ascending = most important first),
greedily assign high precision to the most important layers until the bit
budget is consumed, assign low precision to the rest. This keeps the average
bit-width at or near the target, enabling a fair compression-ratio comparison
between LSAQ and CW-LSAQ.
"""

from __future__ import annotations

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


def allocate_bits(
    scores: Dict[int, float],
    target_avg_bits: float = 5.0,
    precisions: List[int] = None,
) -> Dict[int, int]:
    """Assign bit-widths to layers based on sensitivity scores (lower = more important)."""
    if precisions is None:
        precisions = [8, 4]

    precisions = sorted(precisions, reverse=True)
    high_bits, low_bits = precisions[0], precisions[-1]
    num_layers = len(scores)
    total_budget = target_avg_bits * num_layers

    sorted_layers = sorted(scores, key=lambda idx: scores[idx])

    if high_bits == low_bits:
        n_high = num_layers
    else:
        n_high = (total_budget - num_layers * low_bits) / (high_bits - low_bits)
        n_high = max(0, min(num_layers, round(n_high)))

    allocation = {
        layer_idx: high_bits if i < n_high else low_bits
        for i, layer_idx in enumerate(sorted_layers)
    }

    actual_avg = sum(allocation.values()) / num_layers if num_layers > 0 else 0
    logger.info(
        f"Bit allocation: {n_high}/{num_layers} layers at {high_bits}-bit, "
        f"{num_layers - n_high}/{num_layers} at {low_bits}-bit. "
        f"Actual avg = {actual_avg:.2f} bits (target = {target_avg_bits:.2f})"
    )
    return allocation


def allocation_summary(
    lsaq_alloc: Dict[int, int],
    cwlsaq_alloc: Dict[int, int],
    lsaq_scores: Dict[int, float],
    cwlss_scores: Dict[int, float],
) -> List[Dict]:
    """Side-by-side LSAQ vs CW-LSAQ allocation table."""
    return [
        {
            "layer": idx,
            "lsaq_score": round(lsaq_scores.get(idx, 0), 4),
            "cwlss_score": round(cwlss_scores.get(idx, 0), 4),
            "lsaq_bits": lsaq_alloc[idx],
            "cwlsaq_bits": cwlsaq_alloc[idx],
            "changed": "*" if lsaq_alloc[idx] != cwlsaq_alloc[idx] else "",
        }
        for idx in sorted(lsaq_alloc)
    ]


def estimate_model_size_mb(
    allocation: Dict[int, int],
    params_per_layer: int,
    non_layer_params: int = 0,
    non_layer_bits: int = 16,
) -> float:
    """Estimate compressed model size in MB from per-layer bit-widths."""
    layer_bits = sum(params_per_layer * bits for bits in allocation.values())
    total_bits = layer_bits + non_layer_params * non_layer_bits
    return total_bits / 8 / 1024 / 1024
