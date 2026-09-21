"""
CW-LSAQ Pipeline — Master orchestrator.

This script runs the complete CW-LSAQ pipeline:
  1. Load the base model
  2. Run calibration samples through the model with activation hooks
  3. Compute LSAQ and CW-LSS scores for all layers
  4. Allocate mixed-precision bit-widths under both strategies
  5. QLoRA fine-tune on medical data
  6. Evaluate with the CCER diagnostic probes (4-arm benchmark)
  7. Profile virtual edge feasibility
  8. Generate result tables and comparison outputs

Usage:
    python -m src.main --phase scoring      # Just layer scoring
    python -m src.main --phase full         # Full pipeline (all phases)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import ExperimentConfig, RESULTS_DIR
from src.data.loader import load_medical_qa, split_calibration_eval
from src.quantization.activation_hook import capture_activations
from src.quantization.lsaq_scorer import score_all_layers_lsaq
from src.quantization.cwlss_scorer import score_all_layers_cwlss, compare_scores
from src.quantization.bit_allocator import allocate_bits, allocation_summary, estimate_model_size_mb

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("CW-LSAQ")


def load_model_and_tokenizer(config: ExperimentConfig):
    """Load the base model and tokenizer."""
    logger.info(f"Loading model: {config.base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model_name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def run_layer_scoring(
    model,
    tokenizer,
    config: ExperimentConfig,
    calibration_texts: list[str],
):
    """Run activation capture and compute LSAQ + CW-LSS scores."""
    logger.info(f"Running layer scoring with {len(calibration_texts)} calibration samples...")

    # Get the LM head for projecting activations to vocabulary
    lm_head = model.lm_head

    # capture_activations now takes lm_head & tokenizer so it can
    # project activations to token sets in-hook (per-position, top-k)
    with capture_activations(model, lm_head, tokenizer, k_per_position=config.top_k) as cap:
        for i, text in enumerate(calibration_texts):
            inputs = tokenizer(
                text, return_tensors="pt", truncation=True,
                max_length=config.max_seq_length,
            ).to(model.device)

            with torch.no_grad():
                model(**inputs)

            if (i + 1) % 10 == 0:
                logger.info(f"  Processed {i+1}/{len(calibration_texts)} samples")

        logger.info(f"Captured token sets for {cap.num_layers} layers")

        # Token sets are already computed in-hook
        layer_token_sets = cap.get_layer_token_sets()

    # Compute scores
    logger.info("Computing LSAQ scores (unweighted Jaccard)...")
    lsaq_scores = score_all_layers_lsaq(layer_token_sets)

    logger.info("Computing CW-LSS scores (clinically-weighted Jaccard)...")
    cwlss_scores = score_all_layers_cwlss(layer_token_sets, alpha=config.clinical_token_weight)

    return lsaq_scores, cwlss_scores


def run_bit_allocation(
    lsaq_scores: dict,
    cwlss_scores: dict,
    config: ExperimentConfig,
    model,
):
    """Allocate bits and estimate model sizes."""
    logger.info("Allocating bits under LSAQ strategy...")
    lsaq_alloc = allocate_bits(
        lsaq_scores,
        target_avg_bits=config.target_average_bits,
        precisions=config.supported_precisions,
    )

    logger.info("Allocating bits under CW-LSAQ strategy...")
    cwlsaq_alloc = allocate_bits(
        cwlss_scores,
        target_avg_bits=config.target_average_bits,
        precisions=config.supported_precisions,
    )

    # Estimate model sizes
    total_params = sum(p.numel() for p in model.parameters())
    num_layers = len(lsaq_scores)
    # Rough split: decoder layers hold ~85% of params
    params_per_layer = int(total_params * 0.85 / num_layers)
    non_layer_params = int(total_params * 0.15)

    lsaq_size = estimate_model_size_mb(lsaq_alloc, params_per_layer, non_layer_params)
    cwlsaq_size = estimate_model_size_mb(cwlsaq_alloc, params_per_layer, non_layer_params)
    fp16_size = total_params * 16 / 8 / 1024 / 1024

    summary = allocation_summary(lsaq_alloc, cwlsaq_alloc, lsaq_scores, cwlss_scores)

    return {
        "lsaq_alloc": lsaq_alloc,
        "cwlsaq_alloc": cwlsaq_alloc,
        "summary": summary,
        "sizes": {
            "fp16_mb": round(fp16_size, 2),
            "lsaq_mb": round(lsaq_size, 2),
            "cwlsaq_mb": round(cwlsaq_size, 2),
        },
    }


def save_results(
    lsaq_scores: dict,
    cwlss_scores: dict,
    alloc_results: dict,
    output_dir: Path = RESULTS_DIR,
):
    """Save all results to CSV and JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Score comparison
    comparison = compare_scores(lsaq_scores, cwlss_scores)
    scores_path = output_dir / "layer_scores.csv"
    with open(scores_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Layer", "LSAQ_Score", "CWLSS_Score", "Delta"])
        for idx, data in sorted(comparison.items()):
            writer.writerow([idx, data["lsaq"], data["cwlss"], data["delta"]])
    logger.info(f"Saved layer scores to {scores_path}")

    # Allocation summary
    alloc_path = output_dir / "bit_allocation.csv"
    with open(alloc_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "layer", "lsaq_score", "cwlss_score", "lsaq_bits", "cwlsaq_bits", "changed"
        ])
        writer.writeheader()
        writer.writerows(alloc_results["summary"])
    logger.info(f"Saved bit allocation to {alloc_path}")

    # Model sizes
    sizes_path = output_dir / "model_sizes.json"
    with open(sizes_path, "w") as f:
        json.dump(alloc_results["sizes"], f, indent=2)
    logger.info(f"Saved model sizes to {sizes_path}")

    # Full results JSON
    full_path = output_dir / "full_results.json"
    full_data = {
        "scores": {str(k): v for k, v in comparison.items()},
        "allocation": {
            str(k): {"lsaq": alloc_results["lsaq_alloc"][k], "cwlsaq": alloc_results["cwlsaq_alloc"][k]}
            for k in alloc_results["lsaq_alloc"]
        },
        "sizes": alloc_results["sizes"],
    }
    with open(full_path, "w") as f:
        json.dump(full_data, f, indent=2)
    logger.info(f"Saved full results to {full_path}")


def print_results_table(lsaq_scores, cwlss_scores, alloc_results):
    """Print a formatted results table to stdout."""
    comparison = compare_scores(lsaq_scores, cwlss_scores)

    print("\n" + "=" * 85)
    print("  CW-LSAQ: Layer Sensitivity Scoring Results")
    print("=" * 85)
    header = f"{'Layer':>6} | {'LSAQ':>8} | {'CW-LSS':>8} | {'Delta':>8} | {'LSAQ bits':>9} | {'CW bits':>7} | {'Changed':>7}"
    print(header)
    print("-" * 85)

    for idx in sorted(comparison.keys()):
        d = comparison[idx]
        lsaq_bits = alloc_results["lsaq_alloc"].get(idx, "?")
        cw_bits = alloc_results["cwlsaq_alloc"].get(idx, "?")
        changed = "  *" if lsaq_bits != cw_bits else ""
        print(f"{idx:>6} | {d['lsaq']:>8.4f} | {d['cwlss']:>8.4f} | {d['delta']:>+8.4f} | {lsaq_bits:>9} | {cw_bits:>7} | {changed:>7}")

    print("-" * 85)
    sizes = alloc_results["sizes"]
    print(f"\n  Model Size Estimates:")
    print(f"    FP16 (unquantized):  {sizes['fp16_mb']:>8.1f} MB")
    print(f"    LSAQ quantized:      {sizes['lsaq_mb']:>8.1f} MB")
    print(f"    CW-LSAQ quantized:   {sizes['cwlsaq_mb']:>8.1f} MB")

    # Count changed layers
    n_changed = sum(1 for row in alloc_results["summary"] if row["changed"])
    n_total = len(alloc_results["summary"])
    print(f"\n  Layers with different bit allocation: {n_changed}/{n_total}")
    print("=" * 85 + "\n")


def run_full_pipeline(args, config: ExperimentConfig):
    """Run the complete 4-phase pipeline."""
    from src.data.loader import format_for_finetuning
    from src.models.finetune import run_finetuning, load_finetuned_model
    from src.evaluation.benchmark_4arm import run_4arm_benchmark, print_benchmark_table
    from src.evaluation.virtual_edge import full_edge_profile

    total_start = time.time()

    # ── Phase 1: Layer Scoring ────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("  PHASE 1: Layer Scoring")
    logger.info("=" * 60)

    model, tokenizer = load_model_and_tokenizer(config)

    logger.info("Loading medical QA data...")
    try:
        n_total = args.n_samples + args.n_eval + 20
        records = load_medical_qa(max_samples=n_total)
        calibration, eval_records = split_calibration_eval(records, n_calibration=args.n_samples)
        calibration_texts = [
            f"Question: {r['question']}\nContext: {r['context'][:200]}"
            for r in calibration
        ]
        eval_texts = [
            f"Question: {r['question']}\nContext: {r['context'][:200]}\nAnswer: {r['answer']}"
            for r in eval_records[:args.n_eval]
        ]
        finetune_records = format_for_finetuning(calibration)
    except Exception as e:
        logger.warning(f"Could not load PubMedQA: {e} — using built-in prompts")
        calibration_texts = [
            "What is the recommended starting dose of metformin for type 2 diabetes?",
            "Describe the management of acute myocardial infarction.",
            "A patient presents with no signs of infection after appendectomy.",
            "What are the contraindications for prescribing warfarin?",
            "The lab results show hemoglobin 7.2 g/dL and creatinine 3.5 mg/dL.",
            "Explain the mechanism of action of lisinopril in heart failure.",
            "CT scan shows no evidence of intracranial hemorrhage.",
            "What is the dose of epinephrine in anaphylaxis for adults?",
            "Patient denies chest pain, shortness of breath, or palpitations.",
            "Discuss the use of vancomycin for MRSA bacteremia treatment.",
            "Blood culture results are negative for bacterial growth.",
            "What is the maximum daily dose of acetaminophen in adults?",
            "MRI reveals no ligamentous injury in the right knee.",
            "Prescribe amoxicillin 500mg three times daily for 7 days.",
            "The patient's troponin level is 0.04 ng/mL, within normal limits.",
            "ECG shows no ST-segment elevation or depression.",
            "What is the loading dose of amiodarone for VT?",
            "Assessment: rules out pulmonary embolism based on CT angiography.",
            "Administer furosemide 40mg IV for acute pulmonary edema.",
            "Patient has no known drug allergies (NKDA).",
        ][:args.n_samples]
        eval_texts = calibration_texts[:10]
        finetune_records = [{"text": t} for t in calibration_texts]

    lsaq_scores, cwlss_scores = run_layer_scoring(model, tokenizer, config, calibration_texts)
    alloc_results = run_bit_allocation(lsaq_scores, cwlss_scores, config, model)
    save_results(lsaq_scores, cwlss_scores, alloc_results)
    print_results_table(lsaq_scores, cwlss_scores, alloc_results)

    # ── Phase 2: QLoRA Fine-Tuning ────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("  PHASE 2: QLoRA Fine-Tuning")
    logger.info("=" * 60)

    ft_model, ft_tokenizer = None, None
    ft_results = None

    try:
        ft_results = run_finetuning(config, finetune_records)
        logger.info(f"Fine-tuning complete: loss={ft_results['train_loss']:.4f}, "
                    f"time={ft_results['train_time_sec']:.1f}s")

        ft_path = RESULTS_DIR / "finetune_results.json"
        with open(ft_path, "w") as f:
            json.dump(ft_results, f, indent=2)
        logger.info(f"Fine-tuning results saved to {ft_path}")

        logger.info("Loading fine-tuned (adapter-merged) model for evaluation...")
        ft_model, ft_tokenizer = load_finetuned_model(config)

    except Exception as e:
        logger.error(f"Fine-tuning failed: {e}")
        logger.warning("Proceeding with base model only for benchmarking")

    # ── Phase 3: 4-Arm Benchmark + CCER ──────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("  PHASE 3: CCER Evaluation + 4-Arm Benchmark")
    logger.info("=" * 60)

    benchmark_results = run_4arm_benchmark(
        base_model=model,
        base_tokenizer=tokenizer,
        eval_texts=eval_texts,
        lsaq_alloc=alloc_results["lsaq_alloc"],
        cwlsaq_alloc=alloc_results["cwlsaq_alloc"],
        sizes=alloc_results["sizes"],
        config=config,
        ft_model=ft_model,
        ft_tokenizer=ft_tokenizer,
        output_dir=RESULTS_DIR,
    )
    print_benchmark_table(benchmark_results)

    # ── Phase 4: Virtual Edge Profiling ───────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("  PHASE 4: Virtual Edge Profiling")
    logger.info("=" * 60)

    edge_results = full_edge_profile(model, tokenizer, n_threads=config.edge_cpu_threads)

    print("\n" + "=" * 60)
    print("  Virtual Edge Feasibility (ARM Cortex-A72/A76 simulation)")
    print("=" * 60)
    size_info = edge_results["size"]
    ram_info = edge_results["ram"]
    tput_info = edge_results["throughput"]

    print(f"  Est. model size (5-bit avg): {size_info['total_size_mb']:.1f} MB  "
          f"  [<= 2000 MB: {'YES' if size_info['fits_edge_2gb'] else 'NO'}]")
    print(f"  FP16 baseline size:          {alloc_results['sizes']['fp16_mb']:.1f} MB")
    print(f"  LSAQ quantized size:         {alloc_results['sizes']['lsaq_mb']:.1f} MB")
    print(f"  CW-LSAQ quantized size:      {alloc_results['sizes']['cwlsaq_mb']:.1f} MB")
    print(f"  Peak RAM (FP16 model):       {ram_info['peak_ram_mb']:.1f} MB  "
          f"  [<= 3800 MB: {'YES' if ram_info['fits_edge_4gb'] else 'NO'}]")
    print(f"  CPU throughput ({tput_info['n_threads']} threads): "
          f"{tput_info['avg_tokens_per_sec']:.1f} tok/s  "
          f"(latency: {tput_info['avg_latency_sec']:.2f}s)")
    print(f"  Overall edge feasible:       {'YES' if edge_results['overall_feasible'] else 'NO'}")
    print("=" * 60)

    edge_path = RESULTS_DIR / "edge_profile.json"
    with open(edge_path, "w") as f:
        json.dump(edge_results, f, indent=2, default=str)
    logger.info(f"Edge profile saved to {edge_path}")

    # ── Final Summary ──────────────────────────────────────────────────────
    total_elapsed = time.time() - total_start
    logger.info(f"\nFull pipeline completed in {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    logger.info(f"All results saved to: {RESULTS_DIR}")


def main():
    parser = argparse.ArgumentParser(description="CW-LSAQ Pipeline")
    parser.add_argument(
        "--phase", choices=["scoring", "full"], default="scoring",
        help="Pipeline phase to run",
    )
    parser.add_argument(
        "--n-samples", type=int, default=20,
        help="Number of calibration samples for layer scoring",
    )
    parser.add_argument(
        "--n-eval", type=int, default=50,
        help="Number of evaluation samples for benchmarking (full phase only)",
    )
    args = parser.parse_args()

    config = ExperimentConfig()
    start_time = time.time()

    if args.phase == "full":
        run_full_pipeline(args, config)
        return

    # ── Scoring-only phase ────────────────────────────────────────────────
    model, tokenizer = load_model_and_tokenizer(config)

    logger.info("Loading medical QA calibration data...")
    try:
        records = load_medical_qa(max_samples=args.n_samples + 20)
        calibration, _ = split_calibration_eval(records, n_calibration=args.n_samples)
        calibration_texts = [
            f"Question: {r['question']}\nContext: {r['context'][:200]}"
            for r in calibration
        ]
    except Exception as e:
        logger.warning(f"Could not load PubMedQA dataset: {e}")
        logger.info("Using built-in medical calibration prompts instead...")
        calibration_texts = [
            "What is the recommended starting dose of metformin for type 2 diabetes?",
            "Describe the management of acute myocardial infarction.",
            "A patient presents with no signs of infection after appendectomy.",
            "What are the contraindications for prescribing warfarin?",
            "The lab results show hemoglobin 7.2 g/dL and creatinine 3.5 mg/dL.",
            "Explain the mechanism of action of lisinopril in heart failure.",
            "CT scan shows no evidence of intracranial hemorrhage.",
            "What is the dose of epinephrine in anaphylaxis for adults?",
            "Patient denies chest pain, shortness of breath, or palpitations.",
            "Discuss the use of vancomycin for MRSA bacteremia treatment.",
            "Blood culture results are negative for bacterial growth.",
            "What is the maximum daily dose of acetaminophen in adults?",
            "MRI reveals no ligamentous injury in the right knee.",
            "Prescribe amoxicillin 500mg three times daily for 7 days.",
            "The patient's troponin level is 0.04 ng/mL, within normal limits.",
            "ECG shows no ST-segment elevation or depression.",
            "What is the loading dose of amiodarone for VT?",
            "Assessment: rules out pulmonary embolism based on CT angiography.",
            "Administer furosemide 40mg IV for acute pulmonary edema.",
            "Patient has no known drug allergies (NKDA).",
        ]

    lsaq_scores, cwlss_scores = run_layer_scoring(model, tokenizer, config, calibration_texts)
    alloc_results = run_bit_allocation(lsaq_scores, cwlss_scores, config, model)
    save_results(lsaq_scores, cwlss_scores, alloc_results)
    print_results_table(lsaq_scores, cwlss_scores, alloc_results)

    elapsed = time.time() - start_time
    logger.info(f"Pipeline completed in {elapsed:.1f} seconds")


if __name__ == "__main__":
    main()
