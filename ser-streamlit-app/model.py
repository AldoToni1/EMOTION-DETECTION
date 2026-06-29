"""Definisi arsitektur model WavLM SER dan fungsi pemuatan checkpoint."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import WavLMModel

PRETRAINED_MODEL = "microsoft/wavlm-base-plus"
HIDDEN_SIZE = 768
NUM_LABELS = 6
DEFAULT_DROPOUT = 0.35


class AttentionPooling(nn.Module):
    """Attention pooling dengan MLP (LayerNorm -> Linear -> Tanh -> Dropout -> Linear)."""

    def __init__(self, hidden_size: int = HIDDEN_SIZE, dropout: float = DEFAULT_DROPOUT):
        super().__init__()
        self.attn = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 128),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        scores = self.attn(hidden_states).squeeze(-1)
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        return (hidden_states * weights.unsqueeze(-1)).sum(dim=1)


class WavLMSERModel(nn.Module):
    """Model SER berbasis WavLM dengan attention pooling + mean pooling."""

    def __init__(
        self,
        num_labels: int = NUM_LABELS,
        dropout: float = DEFAULT_DROPOUT,
        pretrained_model: str = PRETRAINED_MODEL,
    ):
        super().__init__()
        self.backbone = WavLMModel.from_pretrained(pretrained_model)
        self.pooling = AttentionPooling(HIDDEN_SIZE, dropout)
        self.classifier = nn.Sequential(
            nn.LayerNorm(HIDDEN_SIZE * 2),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_labels),
        )

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        outputs = self.backbone(input_values, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state

        frame_mask = None
        if attention_mask is not None:
            frame_mask = self.backbone._get_feature_vector_attention_mask(
                hidden_states.shape[1],
                attention_mask,
            )

        attn_pooled = self.pooling(hidden_states, frame_mask)
        if frame_mask is not None:
            mask_expanded = frame_mask.unsqueeze(-1).float()
            mean_pooled = (hidden_states * mask_expanded).sum(dim=1) / mask_expanded.sum(
                dim=1
            ).clamp(min=1e-9)
        else:
            mean_pooled = hidden_states.mean(dim=1)

        features = torch.cat([attn_pooled, mean_pooled], dim=-1)
        return self.classifier(features)


def _strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Hapus prefix 'module.' dari checkpoint DataParallel."""
    if not any(key.startswith("module.") for key in state_dict):
        return state_dict
    return {key.removeprefix("module."): value for key, value in state_dict.items()}


def _extract_state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
    """Ambil state_dict dari berbagai format checkpoint."""
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "model_state", "state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
        tensor_items = {
            key: value
            for key, value in checkpoint.items()
            if isinstance(value, torch.Tensor)
        }
        if tensor_items:
            return tensor_items
        raise ValueError(
            "Checkpoint dictionary tidak berisi state_dict yang dikenali. "
            "Key yang diharapkan: 'model_state_dict', 'model_state', atau 'state_dict'."
        )
    if isinstance(checkpoint, nn.Module):
        return checkpoint.state_dict()
    raise ValueError(
        f"Format checkpoint tidak didukung: {type(checkpoint).__name__}. "
        "Harap gunakan state_dict langsung atau dictionary berisi key model."
    )


def load_model(
    model_path: str,
    device: torch.device | str,
    dropout: float = DEFAULT_DROPOUT,
) -> WavLMSERModel:
    """Buat instance model dan muat bobot dari checkpoint lokal."""
    device = torch.device(device)

    try:
        model = WavLMSERModel(dropout=dropout)
    except Exception as exc:
        raise RuntimeError(
            f"Gagal memuat backbone HuggingFace '{PRETRAINED_MODEL}'. "
            "Pastikan koneksi internet tersedia untuk unduhan pertama kali."
        ) from exc

    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"File model tidak ditemukan: {model_path}"
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"Gagal membaca checkpoint: {model_path}. Error: {exc}"
        ) from exc

    state_dict = _strip_module_prefix(_extract_state_dict(checkpoint))

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        model_keys = set(model.state_dict().keys())
        ckpt_keys = set(state_dict.keys())
        missing = sorted(model_keys - ckpt_keys)
        unexpected = sorted(ckpt_keys - model_keys)
        raise RuntimeError(
            "Checkpoint tidak cocok dengan arsitektur model.\n"
            f"- Key hilang ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}\n"
            f"- Key tidak terduga ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}\n"
            f"Detail: {exc}"
        ) from exc

    model.to(device)
    model.eval()
    return model
