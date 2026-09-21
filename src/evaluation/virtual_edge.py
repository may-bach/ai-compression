"""
Virtual Edge Feasibility Profiler.

Instead of deploying to a physical Raspberry Pi, this module verifies
that the quantized model meets edge-device constraints:

  • Model size on disk   ≤ 2 GB
  • Peak RAM at runtime  ≤ 3.8 GB  (4 GB Pi with OS overhead)
  • CPU-constrained throughput  (4 threads simulating ARM Cortex-A72/A76)
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Dict, Optional

import psutil
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


def profile_model_size(model_path: str | Path) -> Dict[str, float]:
    """Measure the on-disk size of a saved model.

    Parameters
    ----------
    model_path : str or Path
        Directory containing the saved model files.

    Returns
    -------
    dict
        total_size_mb, num_files, fits_edge (bool)
    """
    model_path = Path(model_path)
    total_bytes = 0
    num_files = 0

    if model_path.is_dir():
        for f in model_path.rglob("*"):
            if f.is_file():
                total_bytes += f.stat().st_size
                num_files += 1
    elif model_path.is_file():
        total_bytes = model_path.stat().st_size
        num_files = 1

    size_mb = total_bytes / (1024 * 1024)

    return {
        "total_size_mb": round(size_mb, 2),
        "num_files": num_files,
        "fits_edge_2gb": size_mb <= 2000.0,
    }


def profile_inference_ram(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    sample_prompt: str = "What is the recommended dose of metformin?",
    max_new_tokens: int = 50,
) -> Dict[str, float]:
    """Measure peak RSS memory during CPU inference.

    Returns
    -------
    dict
        baseline_ram_mb, peak_ram_mb, inference_ram_mb, fits_edge_4gb
    """
    process = psutil.Process(os.getpid())

    # Baseline
    baseline_mb = process.memory_info().rss / (1024 * 1024)

    # Run inference
    inputs = tokenizer(sample_prompt, return_tensors="pt", truncation=True, max_length=256)
    inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}

    with torch.no_grad():
        model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    peak_mb = process.memory_info().rss / (1024 * 1024)

    return {
        "baseline_ram_mb": round(baseline_mb, 2),
        "peak_ram_mb": round(peak_mb, 2),
        "inference_ram_mb": round(peak_mb - baseline_mb, 2),
        "fits_edge_4gb": peak_mb <= 3800.0,
    }


def profile_cpu_throughput(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    sample_prompt: str = "What is the recommended dose of metformin for type 2 diabetes?",
    max_new_tokens: int = 50,
    n_threads: int = 4,
    n_runs: int = 3,
) -> Dict[str, float]:
    """Profile inference throughput under constrained CPU threads.

    Simulates ARM Cortex-A72/A76 quad-core by limiting torch threads.

    Returns
    -------
    dict
        avg_tokens_per_sec, avg_latency_sec, total_tokens, n_threads
    """
    original_threads = torch.get_num_threads()
    torch.set_num_threads(n_threads)

    device = next(model.parameters()).device
    inputs = tokenizer(sample_prompt, return_tensors="pt", truncation=True, max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    latencies = []
    total_tokens_generated = 0

    model.eval()
    with torch.no_grad():
        for _ in range(n_runs):
            start = time.perf_counter()
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            elapsed = time.perf_counter() - start

            n_generated = output_ids.shape[1] - inputs["input_ids"].shape[1]
            latencies.append(elapsed)
            total_tokens_generated += n_generated

    torch.set_num_threads(original_threads)

    avg_latency = sum(latencies) / len(latencies)
    avg_tps = (total_tokens_generated / len(latencies)) / avg_latency if avg_latency > 0 else 0

    return {
        "avg_tokens_per_sec": round(avg_tps, 2),
        "avg_latency_sec": round(avg_latency, 3),
        "total_tokens": total_tokens_generated,
        "n_threads": n_threads,
        "n_runs": n_runs,
    }


def full_edge_profile(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    model_path: Optional[str | Path] = None,
    n_threads: int = 4,
) -> Dict:
    """Run all edge feasibility checks.

    Returns
    -------
    dict with keys: size, ram, throughput, overall_feasible
    """
    results: Dict = {}

    # Size check
    if model_path:
        results["size"] = profile_model_size(model_path)
    else:
        # Estimate from parameter count
        total_params = sum(p.numel() for p in model.parameters())
        # Rough estimate: assume average of 5 bits per param for quantized
        estimated_mb = (total_params * 5) / 8 / 1024 / 1024
        results["size"] = {
            "total_size_mb": round(estimated_mb, 2),
            "estimated_from_params": True,
            "fits_edge_2gb": estimated_mb <= 2000.0,
        }

    # RAM check
    results["ram"] = profile_inference_ram(model, tokenizer)

    # Throughput check
    results["throughput"] = profile_cpu_throughput(model, tokenizer, n_threads=n_threads)

    # Overall feasibility
    results["overall_feasible"] = (
        results["size"].get("fits_edge_2gb", True) and
        results["ram"].get("fits_edge_4gb", True)
    )

    return results
