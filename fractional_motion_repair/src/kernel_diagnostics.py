from __future__ import annotations

import torch
from torch import nn

from fractional_models import gl_integral_coefficients


@torch.no_grad()
def kernel_profiles(expert: nn.Module) -> torch.Tensor:
    """Return normalized base-kernel taps in band-by-lag order on the CPU."""
    memory = expert.backbone.memory
    dtype = memory.projection.weight.dtype
    if memory.mode == "instant":
        weights = torch.zeros(1, memory.kernel_size, dtype=dtype)
        weights[:, 0] = 1
    elif memory.mode == "integer":
        weights = torch.zeros(len(memory.finite_windows), memory.kernel_size, dtype=dtype)
        for band, window in enumerate(memory.finite_windows):
            weights[band, :window] = 1 / window
    elif memory.mode == "fractional":
        orders = memory.fractional_orders().detach().cpu()
        if not torch.equal(orders, orders[:1].expand_as(orders)):
            raise ValueError("Revision K3 must have shared fixed orders across joints")
        weights = gl_integral_coefficients(orders[0], memory.kernel_size).T
    else:
        weights = memory.shared_kernel_coefficients().detach().cpu().to(dtype=dtype)
    weights = weights.double()
    if (
        weights.ndim != 2
        or not torch.isfinite(weights).all()
        or (weights < 0).any()
        or (weights.sum(dim=1) <= 0).any()
    ):
        raise ValueError("Cannot summarize invalid temporal kernel weights")
    return weights / weights.sum(dim=1, keepdim=True)


@torch.no_grad()
def kernel_statistics(expert: nn.Module) -> list[dict[str, object]]:
    """Measure each normalized band before visibility normalization and bridging."""
    memory = expert.backbone.memory
    profiles = kernel_profiles(expert)
    lags = torch.arange(profiles.shape[1], dtype=profiles.dtype)
    orders = None
    decays = None
    if memory.mode in {"fractional", "tempered"}:
        orders = memory.fractional_orders().detach().cpu()[0]
    if memory.mode == "tempered":
        decays = memory.raw_decay.detach().cpu().square()
    elif memory.mode == "exponential":
        decays = memory.exponential_decay.detach().cpu()
    return [
        {
            "band": band + 1,
            "mode": memory.mode,
            "kernel_size": profiles.shape[1],
            "positive_taps": int((weights > 0).sum()),
            "mean_lag_frames": float((weights * lags).sum()),
            "zero_lag_mass": float(weights[0]),
            "beta": None if orders is None else float(orders[band]),
            "lambda": None if decays is None else float(decays[band]),
        }
        for band, weights in enumerate(profiles)
    ]
