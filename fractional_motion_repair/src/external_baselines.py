"""PCHIP and PyPOTS SAITS adapters for the motion completion baselines."""

import numpy as np
import torch
from torch import nn


# PyPOTS v1.5 README quick-start architecture and published SAITS loss weights.
SAITS_CONFIG = {
    "pypots_version": "1.5",
    "n_layers": 2,
    "d_model": 256,
    "n_heads": 4,
    "d_k": 64,
    "d_v": 64,
    "d_ffn": 128,
    "dropout": 0.1,
    "attn_dropout": 0.0,
    "diagonal_attention_mask": True,
    "ort_weight": 1.0,
    "mit_weight": 1.0,
}


@torch.no_grad()
def pchip_interpolate(motion: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Interpolate each joint from visible samples, with constant endpoints."""
    from scipy.interpolate import PchipInterpolator

    batch, frames, joints, coords = motion.shape
    working_dtype = torch.float64 if motion.dtype == torch.float64 else torch.float32
    series = (
        motion.to(device="cpu", dtype=working_dtype)
        .transpose(1, 2)
        .reshape(batch * joints, frames, coords)
        .numpy()
    )
    visible = (
        mask[..., 0].to(device="cpu", dtype=torch.bool)
        .transpose(1, 2)
        .reshape(batch * joints, frames)
        .numpy()
    )

    # SciPy can fit independent series jointly when they share the same anchors.
    patterns, groups = np.unique(
        np.packbits(visible, axis=1), axis=0, return_inverse=True,
    )
    filled = np.empty_like(series)
    times = np.arange(frames)
    for group in range(len(patterns)):
        rows = np.flatnonzero(groups == group)
        anchors = np.flatnonzero(visible[rows[0]])
        if anchors.size == 0:
            filled[rows] = 0
            continue

        observations = series[np.ix_(rows, anchors)]
        if anchors.size == 1 or anchors.size == frames:
            filled[rows] = observations
            continue

        start, end = anchors[0], anchors[-1]
        fit = PchipInterpolator(anchors, observations, axis=1, extrapolate=False)
        # One-sided gaps use the nearest visible endpoint, as in linear interpolation.
        filled[rows, :start] = observations[:, :1]
        filled[rows, start : end + 1] = fit(times[start : end + 1])
        filled[rows, end + 1 :] = observations[:, -1:]
        filled[np.ix_(rows, anchors)] = observations

    result = torch.from_numpy(filled.reshape(batch, joints, frames, coords))
    return result.transpose(1, 2).to(device=motion.device, dtype=motion.dtype)


class SaitsCompletion(nn.Module):
    """Use SAITS with the existing pose masks and full training targets."""

    def __init__(self, frames: int = 128, features: int = 51) -> None:
        super().__init__()
        from pypots.nn.modules.saits import BackboneSAITS

        if frames < 2:
            raise ValueError("SAITS diagonal attention requires at least two frames")

        self.backbone = BackboneSAITS(
            n_steps=frames,
            n_features=features,
            n_layers=SAITS_CONFIG["n_layers"],
            d_model=SAITS_CONFIG["d_model"],
            n_heads=SAITS_CONFIG["n_heads"],
            d_k=SAITS_CONFIG["d_k"],
            d_v=SAITS_CONFIG["d_v"],
            d_ffn=SAITS_CONFIG["d_ffn"],
            dropout=SAITS_CONFIG["dropout"],
            attn_dropout=SAITS_CONFIG["attn_dropout"],
        )
        attention_mask = ~torch.eye(frames, dtype=torch.bool)
        self.register_buffer("_attention_mask", attention_mask.unsqueeze(0), persistent=False)

    def _reconstruct(
        self,
        motion: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        target = motion.flatten(2)
        observed = mask.bool().expand_as(motion).flatten(2)
        inputs = torch.where(observed, target, 0.0)
        first, second, final, *_ = self.backbone(
            inputs,
            observed.to(dtype=inputs.dtype),
            self._attention_mask,
        )
        return (first, second, final), observed

    def forward(self, motion: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        (_, _, estimate), _ = self._reconstruct(motion, mask)
        return torch.where(mask.bool(), motion, estimate.reshape_as(motion))

    def training_loss(self, motion: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        reconstructions, observed = self._reconstruct(motion, mask)
        target = motion.flatten(2)
        # PyPOTS v1.5 imputation core averages the observed MAE across all three
        # reconstructions and scores masked training entries on the final one.
        ort = sum(_masked_mae(prediction, target, observed) for prediction in reconstructions) / 3
        mit = _masked_mae(reconstructions[-1], target, ~observed)
        return SAITS_CONFIG["ort_weight"] * ort + SAITS_CONFIG["mit_weight"] * mit


def _masked_mae(
    prediction: torch.Tensor,
    target: torch.Tensor,
    selected: torch.Tensor,
) -> torch.Tensor:
    error = torch.where(selected, (prediction.float() - target.float()).abs(), 0.0)
    return error.sum() / selected.sum().clamp_min(1)
