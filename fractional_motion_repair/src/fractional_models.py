from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from pilot import linear_interpolate


def gl_difference_coefficients(
    alpha: torch.Tensor,
    kernel_size: int,
) -> torch.Tensor:
    """Return truncated Grunwald-Letnikov difference coefficients."""
    if kernel_size < 2:
        raise ValueError("kernel_size must be at least 2")
    coefficients = [torch.ones_like(alpha)]
    for index in range(1, kernel_size):
        previous = coefficients[-1]
        coefficients.append(
            -previous * (alpha - float(index - 1)) / float(index)
        )
    return torch.stack(coefficients)


def _left_filter(signal: torch.Tensor, coefficients: torch.Tensor) -> torch.Tensor:
    channels = signal.shape[1]
    kernel_size = int(coefficients.numel())
    padded = F.pad(signal, (kernel_size - 1, 0), mode="replicate")
    kernel = coefficients.flip(0).view(1, 1, kernel_size).expand(channels, 1, -1)
    return F.conv1d(padded, kernel, groups=channels)


def symmetric_gl_difference(
    signal: torch.Tensor,
    alpha: torch.Tensor,
    kernel_size: int,
) -> torch.Tensor:
    coefficients = gl_difference_coefficients(alpha, kernel_size)
    left = _left_filter(signal, coefficients)
    right = torch.flip(
        _left_filter(torch.flip(signal, dims=(-1,)), coefficients),
        dims=(-1,),
    )
    return 0.5 * (left + right)


def gl_integral_coefficients(
    beta: torch.Tensor,
    kernel_size: int,
) -> torch.Tensor:
    """Return normalized GL fractional-integral coefficients."""
    if kernel_size < 2:
        raise ValueError("kernel_size must be at least 2")
    coefficients = [torch.ones_like(beta)]
    for index in range(1, kernel_size):
        previous = coefficients[-1]
        coefficients.append(
            previous * (beta + float(index - 1)) / float(index)
        )
    stacked = torch.stack(coefficients)
    denominator = stacked.sum(dim=0, keepdim=True)
    return stacked / denominator.clamp_min(torch.finfo(stacked.dtype).eps)


def symmetric_gl_integral_context(
    signal: torch.Tensor,
    beta: torch.Tensor,
    kernel_size: int,
) -> torch.Tensor:
    coefficients = gl_integral_coefficients(beta, kernel_size)
    left = _left_filter(signal, coefficients)
    right = torch.flip(
        _left_filter(torch.flip(signal, dims=(-1,)), coefficients),
        dims=(-1,),
    )
    return 0.5 * (left + right) - signal


def symmetric_finite_window_context(
    signal: torch.Tensor,
    window_size: int,
) -> torch.Tensor:
    if window_size < 3 or window_size % 2 == 0:
        raise ValueError("window_size must be odd and at least 3")
    coefficients = signal.new_full((window_size,), 1.0 / window_size)
    left = _left_filter(signal, coefficients)
    right = torch.flip(
        _left_filter(torch.flip(signal, dims=(-1,)), coefficients),
        dims=(-1,),
    )
    return 0.5 * (left + right) - signal


def _left_masked_average(
    signal: torch.Tensor,
    visibility: torch.Tensor,
    coefficients: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    channels = signal.shape[1]
    kernel_size = int(coefficients.shape[-1])
    if coefficients.ndim == 1:
        kernel = (
            coefficients.flip(0)
            .view(1, 1, kernel_size)
            .expand(channels, 1, -1)
        )
    elif coefficients.shape == (channels, kernel_size):
        kernel = coefficients.flip(-1).unsqueeze(1)
    else:
        raise ValueError("coefficients must be shared or channel-specific")
    weighted = F.pad(signal * visibility, (kernel_size - 1, 0))
    visible = F.pad(visibility, (kernel_size - 1, 0))
    numerator = F.conv1d(weighted, kernel, groups=channels)
    denominator = F.conv1d(visible, kernel, groups=channels)
    return numerator, denominator


def masked_bidirectional_context(
    signal: torch.Tensor,
    visibility: torch.Tensor,
    coefficients: torch.Tensor,
) -> torch.Tensor:
    left_numerator, left_denominator = _left_masked_average(
        signal,
        visibility,
        coefficients,
    )
    right_numerator, right_denominator = _left_masked_average(
        torch.flip(signal, dims=(-1,)),
        torch.flip(visibility, dims=(-1,)),
        coefficients,
    )
    numerator = left_numerator + torch.flip(right_numerator, dims=(-1,))
    denominator = left_denominator + torch.flip(right_denominator, dims=(-1,))
    estimate = numerator / denominator.clamp_min(1e-6)
    return torch.where(denominator > 1e-6, estimate, signal)


def endpoint_constrained_bridge(
    residual_increments: torch.Tensor,
    visibility: torch.Tensor,
) -> torch.Tensor:
    frames = residual_increments.shape[-1]
    observed = visibility.bool()
    indices = torch.arange(
        frames,
        device=residual_increments.device,
        dtype=torch.int64,
    ).view(1, 1, frames)
    previous = torch.where(observed, indices, -torch.ones_like(indices))
    previous = previous.cummax(dim=-1).values
    following = torch.flip(
        torch.flip(
            torch.where(observed, indices, torch.full_like(indices, frames)),
            dims=(-1,),
        ).cummin(dim=-1).values,
        dims=(-1,),
    )
    has_two_endpoints = (previous >= 0) & (following < frames)
    previous_clamped = previous.clamp(0, frames - 1)
    following_clamped = following.clamp(0, frames - 1)
    cumulative = residual_increments.cumsum(dim=-1)
    from_left = cumulative - cumulative.gather(-1, previous_clamped)
    whole_gap = (
        cumulative.gather(-1, following_clamped)
        - cumulative.gather(-1, previous_clamped)
    )
    fraction = (indices - previous).to(residual_increments.dtype) / (
        following - previous
    ).clamp_min(1).to(residual_increments.dtype)
    correction = from_left - fraction * whole_gap
    return torch.where(~observed & has_two_endpoints, correction, 0.0)


class TemporalMemoryAdapter(nn.Module):
    """A parameter-matched fractional or integer long-memory adapter."""

    def __init__(
        self,
        channels: int,
        mode: str,
        learn_orders: bool = True,
        kernel_size: int = 97,
        order_ranges: tuple[tuple[float, float], ...] = (
            (0.10, 0.35),
            (0.35, 0.65),
            (0.65, 0.90),
        ),
        finite_windows: tuple[int, ...] = (9, 33, 97),
    ) -> None:
        super().__init__()
        if mode not in {"fractional", "integer"}:
            raise ValueError(f"Unknown memory mode: {mode}")
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        if len(order_ranges) != len(finite_windows):
            raise ValueError("fractional bands and finite windows must match")
        if max(finite_windows) > kernel_size:
            raise ValueError("finite windows must fit inside kernel_size")

        self.channels = channels
        if channels % 3 != 0:
            raise ValueError("motion-coordinate channels must be divisible by 3")
        self.order_groups = channels // 3
        self.coordinates_per_group = 3
        self.mode = mode
        self.kernel_size = kernel_size
        self.finite_windows = finite_windows
        lower = torch.tensor([item[0] for item in order_ranges])
        upper = torch.tensor([item[1] for item in order_ranges])
        self.register_buffer("order_lower", lower)
        self.register_buffer("order_upper", upper)
        self.raw_orders = nn.Parameter(
            torch.zeros(self.order_groups, len(order_ranges)),
            requires_grad=learn_orders,
        )
        self.mix_logits = nn.Parameter(torch.zeros(channels, len(order_ranges)))
        self.gate_logits = nn.Parameter(torch.full((channels,), -2.0))
        self.projection = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        nn.init.zeros_(self.projection.weight)

    def fractional_orders(self) -> torch.Tensor:
        return self.order_lower[None, :] + (
            self.order_upper - self.order_lower
        )[None, :] * torch.sigmoid(self.raw_orders)

    def _band_features(
        self,
        signal: torch.Tensor,
        visibility: torch.Tensor,
    ) -> torch.Tensor:
        if self.mode == "fractional":
            channel_orders = self.fractional_orders().repeat_interleave(
                self.coordinates_per_group,
                dim=0,
            )
            bands = [
                masked_bidirectional_context(
                    signal,
                    visibility,
                    gl_integral_coefficients(
                        channel_orders[:, band_index],
                        self.kernel_size,
                    ).transpose(0, 1),
                ) - signal
                for band_index in range(channel_orders.shape[1])
            ]
        else:
            bands = [
                masked_bidirectional_context(
                    signal,
                    visibility,
                    signal.new_full((window_size,), 1.0 / window_size),
                ) - signal
                for window_size in self.finite_windows
            ]
        return torch.stack(bands, dim=2)

    def forward(
        self,
        signal: torch.Tensor,
        visibility: torch.Tensor,
    ) -> torch.Tensor:
        increments = torch.zeros_like(signal)
        increments[..., 1:] = signal[..., 1:] - signal[..., :-1]
        increment_visibility = torch.zeros_like(visibility)
        increment_visibility[..., 1:] = (
            visibility[..., 1:] * visibility[..., :-1]
        )
        bands = self._band_features(increments, increment_visibility)
        mixing = torch.softmax(self.mix_logits, dim=-1)
        memory = torch.einsum("bcot,co->bct", bands, mixing)
        gate = torch.sigmoid(self.gate_logits).view(1, -1, 1)
        residual_increments = gate * self.projection(memory)
        correction = endpoint_constrained_bridge(
            residual_increments,
            visibility,
        )
        return signal + correction


class MemoryTemporalConvNet(nn.Module):
    """TCN completion backbone with a power-law or integer memory bank."""

    def __init__(
        self,
        frames: int,
        mode: str,
        learn_orders: bool = True,
        joints: int = 17,
        width: int = 248,
        dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32),
        kernel_size: int = 97,
    ) -> None:
        super().__init__()
        channels = joints * 4
        self.frames = frames
        self.joints = joints
        self.input = nn.Conv1d(channels, width, kernel_size=1)
        self.memory = TemporalMemoryAdapter(
            channels=joints * 3,
            mode=mode,
            learn_orders=learn_orders,
            kernel_size=kernel_size,
        )
        blocks: list[nn.Module] = []
        for dilation in dilations:
            blocks.append(
                nn.Sequential(
                    nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation),
                    nn.GELU(),
                    nn.Conv1d(width, width, 1),
                    nn.GELU(),
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.output = nn.Conv1d(width, joints * 3, kernel_size=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, motion: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        initial = linear_interpolate(motion, mask)
        motion_signal = initial.flatten(2).transpose(1, 2)
        visibility = mask.expand_as(initial).flatten(2).transpose(1, 2)
        adjusted_signal = self.memory(motion_signal, visibility)
        adjusted = adjusted_signal.transpose(1, 2).reshape_as(initial)
        features = torch.cat((adjusted, mask), dim=-1).flatten(2).transpose(1, 2)
        hidden = self.input(features)
        for block in self.blocks:
            hidden = hidden + block(hidden)
        residual = self.output(hidden).transpose(1, 2).reshape_as(initial)
        prediction = initial + residual
        return torch.where(mask.bool(), motion, prediction)


def fractional_trajectory_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    alpha: float = 0.5,
    kernel_size: int = 5,
) -> torch.Tensor:
    """Fractional trajectory discrepancy on missing joint coordinates."""
    batch, frames, joints, coordinates = prediction.shape
    pred_signal = prediction.permute(0, 2, 3, 1).reshape(
        batch, joints * coordinates, frames
    )
    target_signal = target.permute(0, 2, 3, 1).reshape(
        batch, joints * coordinates, frames
    )
    order = prediction.new_tensor(alpha)
    pred_difference = symmetric_gl_difference(pred_signal, order, kernel_size)
    target_difference = symmetric_gl_difference(target_signal, order, kernel_size)
    distance = (pred_difference - target_difference).reshape(
        batch, joints, coordinates, frames
    ).permute(0, 3, 1, 2)
    distance = torch.linalg.vector_norm(distance, dim=-1)
    missing = 1.0 - mask[..., 0]
    return (distance * missing).sum() / missing.sum().clamp_min(1.0)


def memory_parameter_count(channels: int = 51) -> int:
    # Projection + three-band mixing + channel gates + three learned orders.
    return channels * channels + channels * 5


def initial_order_values() -> tuple[float, ...]:
    ranges = ((0.10, 0.35), (0.35, 0.65), (0.65, 0.90))
    return tuple((lower + upper) / 2.0 for lower, upper in ranges)


def validate_fractional_definition() -> None:
    alpha = torch.tensor(0.5)
    coefficients = gl_difference_coefficients(alpha, 5)
    expected = torch.tensor([1.0, -0.5, -0.125, -0.0625, -0.0390625])
    if not torch.allclose(coefficients, expected, atol=1e-7):
        raise RuntimeError("Grunwald-Letnikov recurrence validation failed")
    if not all(math.isfinite(value) for value in coefficients.tolist()):
        raise RuntimeError("Non-finite fractional coefficient")
    integral = gl_integral_coefficients(alpha, 5)
    expected_integral = torch.tensor([1.0, 0.5, 0.375, 0.3125, 0.2734375])
    expected_integral = expected_integral / expected_integral.sum()
    if not torch.allclose(integral, expected_integral, atol=1e-7):
        raise RuntimeError("GL fractional-integral recurrence validation failed")
