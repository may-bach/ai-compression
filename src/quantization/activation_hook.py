"""
Activation Hook Engine — captures per-layer token predictions for layer profiling.

For each calibration forward pass, hooks project the input and output hidden
states of every decoder layer through the LM head at every sequence position,
take the top-k most likely tokens per position, and union them into per-sample
token sets A_l and B_l used by the scorers.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Dict, List, Set, Tuple

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


class LayerActivationCapture:
    """Captures per-position token predictions at each decoder layer."""

    def __init__(
        self,
        lm_head: nn.Linear,
        tokenizer: PreTrainedTokenizerBase,
        k_per_position: int = 3,
    ) -> None:
        self.lm_head = lm_head
        self.tokenizer = tokenizer
        self.k_per_position = k_per_position
        self.input_token_sets: Dict[int, List[List[str]]] = {}
        self.output_token_sets: Dict[int, List[List[str]]] = {}
        self._hooks: List[torch.utils.hooks.RemovableHook] = []

    def _project_to_token_set(self, hidden: torch.Tensor) -> List[str]:
        """Project (seq_len, hidden_dim) hidden states to union of top-k token strings."""
        logits = self.lm_head(hidden.to(dtype=self.lm_head.weight.dtype)).float()
        top_ids = torch.topk(logits, self.k_per_position, dim=-1).indices
        unique_ids = set(top_ids.reshape(-1).cpu().tolist())
        return [t for tid in unique_ids if (t := self.tokenizer.decode([tid]).strip().lower())]

    def _make_hook(self, layer_idx: int):
        def hook_fn(module: nn.Module, inp, out):
            with torch.no_grad():
                inp_h = (inp[0] if isinstance(inp, tuple) else inp).detach()[0]
                out_h = (out[0] if isinstance(out, tuple) else out).detach()[0]
            self.input_token_sets.setdefault(layer_idx, []).append(
                self._project_to_token_set(inp_h)
            )
            self.output_token_sets.setdefault(layer_idx, []).append(
                self._project_to_token_set(out_h)
            )
        return hook_fn

    def attach(self, model: PreTrainedModel) -> None:
        self.clear()
        layers = _get_decoder_layers(model)
        logger.info(f"Attaching activation hooks to {len(layers)} decoder layers")
        for idx, layer in enumerate(layers):
            self._hooks.append(layer.register_forward_hook(self._make_hook(idx)))

    def detach(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def clear(self) -> None:
        self.detach()
        self.input_token_sets.clear()
        self.output_token_sets.clear()

    @property
    def num_layers(self) -> int:
        return len(self.input_token_sets)

    def get_layer_token_sets(self) -> Dict[int, Tuple[List[List[str]], List[List[str]]]]:
        """Return layer_idx → (input_token_sets, output_token_sets)."""
        return {
            idx: (self.input_token_sets[idx], self.output_token_sets[idx])
            for idx in sorted(self.input_token_sets)
        }


@contextmanager
def capture_activations(
    model: PreTrainedModel,
    lm_head: nn.Linear,
    tokenizer: PreTrainedTokenizerBase,
    k_per_position: int = 3,
):
    """Context manager that yields a LayerActivationCapture attached to model."""
    cap = LayerActivationCapture(lm_head, tokenizer, k_per_position)
    cap.attach(model)
    try:
        yield cap
    finally:
        cap.detach()


def _get_decoder_layers(model: PreTrainedModel) -> nn.ModuleList:
    """Extract decoder layer list — supports Llama, Qwen, Mistral, Phi, Gemma."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    candidates = [
        (name, mod) for name, mod in model.named_modules()
        if isinstance(mod, nn.ModuleList) and len(mod) > 4
    ]
    if candidates:
        best = max(candidates, key=lambda x: len(x[1]))
        logger.info(f"Found decoder layers at '{best[0]}' ({len(best[1])} layers)")
        return best[1]
    raise ValueError("Could not locate decoder layers in model architecture")
