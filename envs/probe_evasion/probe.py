"""Probe loading, activation extraction, and inference for probe evasion environment."""

from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F


class ActivationCaptureHook:
    """Context manager for capturing residual stream activations at a target layer.

    Adapted from envs/ssc/ssc_act_tokens/extract_similarities.py:ResidualCaptureHook.
    """

    def __init__(self, model, target_layer: int):
        self.model = model
        self.target_layer = target_layer
        self.representations = None
        self.hook_handle = None

    def _capture_hook(self, module, input, output):
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output
        self.representations = hidden_states.detach().clone()

    def __enter__(self):
        target_layer_module = self.model.model.layers[self.target_layer]
        self.hook_handle = target_layer_module.register_forward_hook(self._capture_hook)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.hook_handle is not None:
            self.hook_handle.remove()


class LinearProbe:
    """Wrapper for a trained nn.Linear probe on residual stream activations.

    Expected saved format: a state dict with 'weight' and optionally 'bias' keys,
    from an nn.Linear(hidden_dim, num_classes) or nn.Linear(hidden_dim, 1).
    """

    def __init__(self, weights_path: str, threshold: float = 0.5, device: str = "cpu"):
        self.threshold = threshold
        self.device = device
        self.probe = self._load_probe(weights_path)

    def _load_probe(self, weights_path: str) -> torch.nn.Linear:
        state_dict = torch.load(weights_path, map_location=self.device, weights_only=True)

        # Infer dimensions from weight shape
        if "weight" in state_dict:
            out_features, in_features = state_dict["weight"].shape
        else:
            raise ValueError(f"Expected 'weight' key in state dict, got: {list(state_dict.keys())}")

        has_bias = "bias" in state_dict
        probe = torch.nn.Linear(in_features, out_features, bias=has_bias)
        probe.load_state_dict(state_dict)
        probe.to(self.device)
        probe.eval()
        return probe

    def predict(self, activations: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """Run probe on activations.

        Args:
            activations: Tensor of shape (batch, hidden_dim)

        Returns:
            (predictions, probabilities) — both numpy arrays of shape (batch,).
            predictions: binary (0 or 1). probabilities: float in [0, 1].
        """
        with torch.no_grad():
            logits = self.probe(activations.to(self.device))

            if logits.shape[-1] == 1:
                # Binary probe: single output, use sigmoid
                probs = torch.sigmoid(logits).squeeze(-1)
            elif logits.shape[-1] == 2:
                # Binary probe: two outputs, use softmax on class 1
                probs = F.softmax(logits, dim=-1)[:, 1]
            else:
                # Multi-class: take max probability
                probs = F.softmax(logits, dim=-1).max(dim=-1).values

        probs_np = probs.cpu().numpy()
        preds_np = (probs_np >= self.threshold).astype(int)
        return preds_np, probs_np


def extract_activations(
    model,
    tokenizer,
    formatted_prompts: List[str],
    layer: int,
    pooling: str = "last_token",
    device: str = "cuda",
) -> torch.Tensor:
    """Extract activations from a specific layer for a list of formatted prompts.

    Runs a single forward pass per prompt (no generation) with a hook to capture
    the residual stream at the target layer.

    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        formatted_prompts: List of prompts already formatted with chat template
        layer: Transformer layer index to capture
        pooling: How to aggregate across sequence positions.
            "last_token": use the last non-padding token's activation
            "mean": mean-pool across all non-padding tokens
        device: Device for computation

    Returns:
        Tensor of shape (len(formatted_prompts), hidden_dim)
    """
    all_activations = []

    for prompt in formatted_prompts:
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=False,
            add_special_tokens=False,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        seq_len = inputs["input_ids"].shape[1]

        with ActivationCaptureHook(model, layer) as hook:
            with torch.no_grad():
                model(**inputs)
            # hook.representations: (1, seq_len, hidden_dim)
            reps = hook.representations.squeeze(0)  # (seq_len, hidden_dim)

        if pooling == "last_token":
            # Use the last token's activation
            pooled = reps[seq_len - 1]  # (hidden_dim,)
        elif pooling == "mean":
            pooled = reps.mean(dim=0)  # (hidden_dim,)
        else:
            raise ValueError(f"Unknown pooling strategy: {pooling}. Use 'last_token' or 'mean'.")

        all_activations.append(pooled.cpu())

    return torch.stack(all_activations)  # (batch, hidden_dim)
