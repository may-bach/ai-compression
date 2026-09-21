"""
Dataset loader — calibration & evaluation data for CW-LSAQ.

Loads a medical QA dataset from Hugging Face for:
  • Fine-tuning calibration  (medical question–answer pairs)
  • Layer activation profiling (calibration prompts driven through the model)
  • Task-accuracy evaluation  (held-out QA split)

Default dataset: a subset of PubMedQA (pqa_labeled split).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from datasets import load_dataset

logger = logging.getLogger(__name__)


def load_medical_qa(
    dataset_name: str = "qiaojin/PubMedQA",
    subset: str = "pqa_labeled",
    split: str = "train",
    max_samples: Optional[int] = None,
) -> List[Dict[str, str]]:
    """Load a medical QA dataset and return list of {question, context, answer} dicts.

    Parameters
    ----------
    dataset_name : str
        Hugging Face dataset identifier.
    subset : str
        Dataset subset / configuration name.
    split : str
        Dataset split.
    max_samples : int, optional
        Limit the number of samples returned.

    Returns
    -------
    list of dict
        Each dict has keys: "question", "context", "answer".
    """
    logger.info(f"Loading dataset {dataset_name}/{subset} split={split}")
    ds = load_dataset(dataset_name, subset, split=split)

    records: List[Dict[str, str]] = []
    for i, row in enumerate(ds):
        if max_samples and i >= max_samples:
            break

        # PubMedQA structure: question, context (dict with contexts list), long_answer, final_decision
        question = row.get("question", "")
        context_data = row.get("context", {})

        # Context can be a dict with 'contexts' key (list of strings)
        if isinstance(context_data, dict):
            contexts = context_data.get("contexts", [])
            context_str = " ".join(contexts) if contexts else ""
        elif isinstance(context_data, str):
            context_str = context_data
        elif isinstance(context_data, list):
            context_str = " ".join(context_data)
        else:
            context_str = str(context_data) if context_data else ""

        answer = row.get("long_answer", row.get("final_decision", ""))

        records.append({
            "question": question,
            "context": context_str,
            "answer": str(answer),
        })

    logger.info(f"Loaded {len(records)} samples from {dataset_name}/{subset}")
    return records


def format_for_finetuning(
    records: List[Dict[str, str]],
    max_context_len: int = 300,
) -> List[Dict[str, str]]:
    """Format records into instruction-tuning style text pairs.

    Returns a list of dicts with keys "text" suitable for causal LM training.
    """
    formatted: List[Dict[str, str]] = []
    for rec in records:
        ctx = rec["context"][:max_context_len] if rec["context"] else ""
        prompt = f"Question: {rec['question']}"
        if ctx:
            prompt = f"Context: {ctx}\n\n{prompt}"

        text = f"{prompt}\n\nAnswer: {rec['answer']}"
        formatted.append({"text": text})

    return formatted


def split_calibration_eval(
    records: List[Dict[str, str]],
    n_calibration: int = 50,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """Split records into calibration (for activation profiling) and eval sets."""
    calibration = records[:n_calibration]
    evaluation = records[n_calibration:]
    return calibration, evaluation
