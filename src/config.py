"""Configuration for CW-LSAQ (Clinically-Aware Layer-Specific Adaptive Quantization)."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "saved_models"

for _d in [DATA_DIR, RESULTS_DIR, MODELS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)


@dataclass
class ExperimentConfig:
    """Central experiment configuration."""

    # Model
    base_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    adapter_name: str = "medical_qlora_adapter"

    # Quantization
    supported_precisions: List[int] = field(default_factory=lambda: [8, 4])
    target_average_bits: float = 5.0

    # CW-LSS
    clinical_token_weight: float = 5.0  # w(t) for clinically salient tokens
    top_k: int = 3                      # top-k tokens per sequence position

    # QLoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    learning_rate: float = 2e-4
    num_train_epochs: int = 1
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 512

    # Virtual edge constraints (Raspberry Pi 4 class)
    edge_cpu_threads: int = 4
    edge_max_ram_mb: float = 3800.0
    edge_max_size_mb: float = 2000.0

    # Evaluation
    num_calibration_samples: int = 50
    num_eval_samples: int = 100
