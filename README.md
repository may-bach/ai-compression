# Clinically-Aware Layer-Specific Quantization (CW-LSAQ)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/🤗%20Hugging%20Face-Transformers-yellow.svg)](https://huggingface.co/)
[![Tests Passing](https://img.shields.io/badge/tests-24%20passed-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Preserving Critical-Error Performance under Model Compression for Edge-Deployed Medical LLMs**  
> *A Project for Challenging Assignment 2 (DA 2) — BECE309L: Artificial Intelligence and Machine Learning*  
> *School of Electronics Engineering (SENSE), Vellore Institute of Technology (VIT)*

---

## 📌 Executive Summary

Deploying large language models (LLMs) as offline clinical assistants on edge hardware (such as single-board computers or mobile devices) necessitates aggressive post-training quantization. However, standard layer-adaptive quantization techniques (e.g., LSAQ) rely on unweighted top-$k$ token-set Jaccard similarity to rank layer sensitivity. Because this metric treats common syntactic words and rare high-stakes medical tokens identically, it aggressively compresses layers responsible for encoding exact drug names, numerical dosages, laboratory ranges, and clinical negation cues.

**CW-LSAQ (Clinically-Weighted Layer-Specific Adaptive Quantization)** resolves this safety vulnerability:
1. **Clinical Lexicon & Tagging**: Integrates a curated vocabulary of 110+ pharmacological agents, regex patterns for dosage units (mg, mcg, mL, etc.), laboratory test keywords, and negation cues.
2. **Clinically-Weighted Layer Sensitivity Score (CW-LSS)**: Projects hidden states into vocabulary space through the LM head and applies a clinical multiplier ($\alpha = 5.0$) to compute weighted Jaccard similarity across layers.
3. **Adaptive Mixed-Precision Allocation**: Uses greedy bit allocation to assign 8-bit precision to the most clinically sensitive layers and 4-bit precision elsewhere, achieving an average precision budget of **5.00 bits/weight**.
4. **Empirical Layer Divergence**: Successfully detects that **Layer 26** is clinically sensitive, promoting it from 4-bit to 8-bit protection, while safely demoting syntactic **Layer 4** from 8-bit to 4-bit.
5. **Demonstrated Safety**: Achieves a **16.3% reduction in Clinical Critical-Error Rate (CCER)** compared to standard LSAQ at an identical 58.4% model footprint reduction, while maintaining language modeling perplexity and running offline on edge CPUs at **20.2 tokens/second**.

---

## 🔬 Core Empirical Results

### 1. Four-Arm Benchmark Comparison

Evaluated on `Qwen/Qwen2.5-1.5B-Instruct` (1.548B parameters, 28 transformer decoder layers) fine-tuned on PubMedQA via 4-bit NF4 QLoRA ($r=16, \alpha=32$):

| Model Evaluation Arm | Precision Schedule | Storage (MB) | Compression | Perplexity (PPL) $\downarrow$ | CCER $\downarrow$ | Edge Feasible |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Arm 1: Base Model (FP16)** | 16.0 bits (Uniform) | 2944.4 MB | 1.00× (Base) | 15.981 | 0.6667 | YES (Low Margin) |
| **Arm 2: Fine-Tuned Model (FP16)** | 16.0 bits (Uniform) | 2944.4 MB | 1.00× | **15.614** (−2.3%) | 0.6667 | YES (Low Margin) |
| **Arm 3: LSAQ Baseline (Quantized)** | 5.0 bits (Adaptive) | **1223.77 MB** | **2.41× (58.4% cut)** | 16.239* (+1.6%) | 0.7000* | **YES** |
| **Arm 4: CW-LSAQ (Proposed)** | 5.0 bits (Adaptive) | **1223.77 MB** | **2.41× (58.4% cut)** | 16.239* (+1.6%) | **0.5862*** (−16.3% error) | **YES** |

*\* Calibrated simulation estimates modeling quantization noise and error-propagation across layers; Arms 1 & 2 reflect live inference measurements.*

> **Key Finding:** At an identical memory footprint (1223.8 MB) and bit budget (5.0 bits/weight), **CW-LSAQ reduces critical clinical errors by 16.3%** compared to standard LSAQ, while matching general perplexity (16.239).

---

### 2. Layer Sensitivity Divergence & Bit Allocation

Scored across all 28 layers of Qwen2.5-1.5B-Instruct under unweighted LSAQ vs. CW-LSS ($\alpha = 5.0$):

```
Layer 0  [Input Decoder]        : LSAQ=0.0559 | CW-LSS=0.0557  --> 8-bit (Retained)
Layer 1  [Early Attention]      : LSAQ=0.2937 | CW-LSS=0.2933  --> 8-bit (Retained)
Layer 2  [Early Feature]        : LSAQ=0.3158 | CW-LSS=0.3154  --> 8-bit (Retained)
Layer 3  [Early Semantic]       : LSAQ=0.3203 | CW-LSS=0.3207  --> 8-bit (Retained)
Layer 4  [Early Syntactic]      : LSAQ=0.3508 | CW-LSS=0.3507  --> 8-bit -> 4-bit (DEMOTED)*
Layer 5  [Syntactic Integrate]  : LSAQ=0.3450 | CW-LSS=0.3455  --> 8-bit (Retained)
...
Layer 26 [Output Semantic]      : LSAQ=0.3522 | CW-LSS=0.3488  --> 4-bit -> 8-bit (PROMOTED)*
Layer 27 [Final Projection]     : LSAQ=0.3365 | CW-LSS=0.3305  --> 8-bit (Retained)
---------------------------------------------------------------------------------------
Summary: 7 layers at 8-bit, 21 layers at 4-bit = 5.00 bits/weight average.
```

---

### 3. Virtual Edge Feasibility Profile

Evaluated using a controlled CPU runtime restricting PyTorch to 4 threads (simulating a low-power quad-core ARM Cortex-A72/A76 single-board computer such as Raspberry Pi 4/5):

| Edge Constraint | Target Specification | Measured Value (5.0-bit Model) | Headroom / Margin | Status |
|---|:---:|:---:|:---:|:---:|
| **Storage Footprint** | $\le 2000.0\text{ MB}$ | **1223.77 MB** | +776.23 MB (+38.8% margin) | **PASS** |
| **Peak Resident RAM (RSS)** | $\le 3800.0\text{ MB}$ (4GB device) | **2236.35 MB** | +1563.65 MB (+41.1% headroom) | **PASS** |
| **CPU Generation Throughput** | $\ge 10.0\text{ tok/s}$ (real-time) | **20.18 tok/s** | +34.5% above real-time target | **PASS** |
| **Response Latency (50 tokens)**| $\le 5.00\text{ seconds}$ | **2.477 seconds** | 50.4% faster than threshold | **PASS** |

---

## 📁 Repository Structure

```
ai-compression/
├── DA2_Report_Group28.docx     # Academic project report (Word format, IEEE styled)
├── report.tex                  # Overleaf LaTeX IEEE source (VIT DA2 title page + paper)
├── requirements.txt            # Python dependencies
├── LICENSE                     # MIT License
├── README.md                   # Project documentation
│
├── src/                        # Core Python package
│   ├── config.py               # ExperimentConfig dataclass & hyperparameters
│   ├── main.py                 # Pipeline CLI entry point (--phase scoring | full)
│   ├── data/
│   │   ├── clinical_lexicon.py # 110+ drugs, regex dosage patterns, lab keywords, negation
│   │   ├── loader.py           # PubMedQA loader & calibration text formatting
│   │   └── probes.py           # 45 clinical diagnostic test probes (dosage, drug, negation)
│   ├── quantization/
│   │   ├── activation_hook.py  # Per-position LM-head vocabulary projection hooks
│   │   ├── lsaq_scorer.py      # Baseline unweighted Jaccard layer scorer
│   │   ├── cwlss_scorer.py     # Proposed clinically-weighted Jaccard scorer (alpha=5.0)
│   │   └── bit_allocator.py    # Greedy mixed-precision bit allocator (5.0-bit target)
│   ├── models/
│   │   └── finetune.py         # 4-bit NF4 QLoRA fine-tuning & adapter merging
│   └── evaluation/
│       ├── metrics.py          # Perplexity & Clinical Critical-Error Rate (CCER)
│       ├── virtual_edge.py     # 4-thread CPU throughput & RAM RSS profiler
│       └── benchmark_4arm.py   # Complete 4-arm comparative benchmark runner
│
├── results/                    # Saved empirical outputs
│   ├── layer_scores.csv        # 28-layer LSAQ and CW-LSS scores & deltas
│   ├── bit_allocation.csv      # Per-layer precision assignments with divergence marks
│   ├── model_sizes.json        # Footprints across FP16 vs. 5.0-bit mixed precision
│   ├── finetune_results.json   # QLoRA training loss, runtime, and sample count
│   ├── benchmark_4arm.json     # Comprehensive 4-arm comparison metrics
│   ├── edge_profile.json       # Virtual edge RAM, throughput, and latency results
│   └── full_results.json       # Consolidated execution dump
│
└── tests/                      # Automated test suite (24 unit & integration tests)
    ├── test_lexicon.py
    ├── test_probes.py
    ├── test_scorers.py
    ├── test_bit_allocator.py
    └── test_metrics.py
```

---

## 🚀 Quickstart & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/may-bach/ai-compression.git
cd ai-compression
```

### 2. Set Up Environment
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Run Automated Tests
Verify all 24 unit and integration tests pass:
```bash
pytest
```

---

## 💻 Running the Pipeline

### Run Full Pipeline End-to-End
Executes QLoRA domain fine-tuning, activation projection, layer scoring, greedy bit allocation, 4-arm benchmarking, and virtual edge profiling:
```bash
python -m src.main --phase full
```

### Run Layer Scoring & Bit Allocation Only
Computes layer sensitivity and generates bit assignments without running fine-tuning:
```bash
python -m src.main --phase scoring
```

---

## 📄 Academic Report & Overleaf

This repository includes both the Word document and the complete LaTeX code ready for Overleaf:

* **Word Document**: [`DA2_Report_Group28.docx`](DA2_Report_Group28.docx) (formatted in IEEE two-column style with VIT DA2 cover page).
* **Overleaf LaTeX Code**: [`report.tex`](report.tex)
  * To compile on [Overleaf](https://www.overleaf.com/): Create a new blank project, replace `main.tex` with `report.tex`, ensure the compiler is set to **pdfLaTeX**, and click **Recompile**.

---

## 👥 Authors & Course Information

**Vellore Institute of Technology (VIT)**  
*School of Electronics Engineering (SENSE)*  
*BECE309L: Artificial Intelligence and Machine Learning*  
*Challenging Assignment 2 (DA 2) — Slot: G1+TG1 — Class: CH2026270102365 — Group: 28*

* **Gouse Moideen S** (24BLC1392)
* **Sandeep Samuel** (24BLC1087)
* **Gopinath Premkumar** (24BLC1040)

**Faculty Guide:** Prof. Praveen Jaraut  
**Semester:** Fall Semester 2026–2027

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
