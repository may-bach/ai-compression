"""
Evaluation Metrics — perplexity, task accuracy, and Clinical Critical-Error Rate (CCER).

CCER is the project's key safety metric:

    CCER = (E_dosage + E_drug + E_negation) / Total_Probes

It specifically counts errors on clinically decisive tokens that aggregate
metrics like perplexity would miss because they are diluted across a large
evaluation set.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from src.data.probes import ALL_PROBES, ClinicalProbe, get_probes_by_category

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════
#  Perplexity
# ════════════════════════════════════════════════════════════════════════

def compute_perplexity(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    texts: List[str],
    max_length: int = 512,
    device: Optional[str] = None,
) -> float:
    """Compute perplexity of model on a list of texts.

    Returns
    -------
    float
        Perplexity (lower is better).
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for text in texts:
            encodings = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            ).to(device)

            input_ids = encodings["input_ids"]
            if input_ids.size(1) < 2:
                continue

            outputs = model(**encodings, labels=input_ids)
            loss = outputs.loss
            n_tokens = input_ids.size(1) - 1  # next-token prediction

            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens

    if total_tokens == 0:
        return float("inf")

    avg_loss = total_loss / total_tokens
    return math.exp(avg_loss)


# ════════════════════════════════════════════════════════════════════════
#  Clinical Critical-Error Rate (CCER)
# ════════════════════════════════════════════════════════════════════════

def evaluate_probe(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    probe: ClinicalProbe,
    max_new_tokens: int = 150,
    device: Optional[str] = None,
) -> Dict:
    """Run a single clinical probe and check for errors.

    Returns
    -------
    dict with keys:
        category, prompt, generated, passed,
        expected_found, forbidden_found
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    # Format as a chat-style prompt
    messages = [{"role": "user", "content": probe.prompt}]

    # Try to use chat template if available
    try:
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        input_text = f"Question: {probe.prompt}\nAnswer:"

    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Deterministic greedy for reproducibility
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the generated portion
    generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    gen_lower = generated_text.lower()

    # Check expected tokens (at least one must be present)
    expected_found = []
    for tok in probe.expected_tokens:
        if tok.lower() in gen_lower:
            expected_found.append(tok)

    # Check forbidden tokens (none should be present)
    forbidden_found = []
    for tok in probe.forbidden_tokens:
        if tok.lower() in gen_lower:
            forbidden_found.append(tok)

    # Pass = at least one expected token present AND no forbidden tokens
    passed = (len(expected_found) > 0) and (len(forbidden_found) == 0)

    return {
        "category": probe.category,
        "prompt": probe.prompt[:80] + "...",
        "generated": generated_text[:200],
        "passed": passed,
        "expected_found": expected_found,
        "forbidden_found": forbidden_found,
    }


def compute_ccer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    probes: Optional[List[ClinicalProbe]] = None,
    device: Optional[str] = None,
) -> Dict:
    """Compute Clinical Critical-Error Rate (CCER).

    Returns
    -------
    dict with keys:
        ccer, ccer_dosage, ccer_drug, ccer_negation,
        total_probes, total_errors, per_category, details
    """
    if probes is None:
        probes = ALL_PROBES

    results = []
    for probe in probes:
        result = evaluate_probe(model, tokenizer, probe, device=device)
        results.append(result)

    # Aggregate by category
    categories = {"dosage": [], "drug": [], "negation": []}
    for r in results:
        cat = r["category"]
        if cat in categories:
            categories[cat].append(r)

    per_category = {}
    total_errors = 0
    total_probes = len(results)

    for cat, cat_results in categories.items():
        n_probes = len(cat_results)
        n_errors = sum(1 for r in cat_results if not r["passed"])
        total_errors += n_errors
        per_category[cat] = {
            "probes": n_probes,
            "errors": n_errors,
            "error_rate": n_errors / n_probes if n_probes > 0 else 0.0,
        }

    ccer = total_errors / total_probes if total_probes > 0 else 0.0

    return {
        "ccer": round(ccer, 4),
        "total_probes": total_probes,
        "total_errors": total_errors,
        "per_category": per_category,
        "details": results,
    }
