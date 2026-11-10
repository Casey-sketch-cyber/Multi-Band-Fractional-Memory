from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


BODY_EDGES = (
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (0, 5),
    (0, 6),
)

LIMB_GROUPS = (
    (5, 7, 9),
    (6, 8, 10),
    (11, 13, 15),
    (12, 14, 16),
    (5, 6, 7, 8, 9, 10),
    (11, 12, 13, 14, 15, 16),
)

BODY_PARTS = (
    (0, 1, 2, 3, 4),
    (5, 6, 11, 12),
    (7, 9),
    (8, 10),
    (13, 15),
    (14, 16),
)


@dataclass(frozen=True)
class PilotConfig:
    seed: int = 20260815
    frames: int = 128
    batch_size: int = 32
    epochs: int = 18
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    hankel_window: int = 12
    hankel_rank: int = 4
    position_weight: float = 1.0
    velocity_weight: float = 0.15
    bone_weight: float = 0.10


class MotionDataset(Dataset[torch.Tensor]):
    def __init__(self, motions: np.ndarray, indices: np.ndarray) -> None:
        self.motions = motions
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> torch.Tensor:
        motion = np.asarray(self.motions[self.indices[index]], dtype=np.float32)
        return torch.from_numpy(motion)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def choreography_group(name: str) -> tuple[str, str]:
    genre = name.split("_", maxsplit=1)[0]
    match = re.search(r"_ch(\d+)", name)
    choreography = f"ch{match.group(1)}" if match else name
    return genre, f"{genre}_{choreography}"


def grouped_split(names: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    by_genre: dict[str, list[str]] = {}
    group_for_index: list[str] = []
    for raw_name in names.astype(str):
        genre, group = choreography_group(raw_name)
        by_genre.setdefault(genre, []).append(group)
        group_for_index.append(group)

    split_groups: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    for genre, repeated_groups in sorted(by_genre.items()):
        groups = sorted(set(repeated_groups))
        genre_seed = int(hashlib.sha1(f"{seed}:{genre}".encode()).hexdigest()[:8], 16)
        rng = random.Random(genre_seed)
        rng.shuffle(groups)
        n_test = max(1, round(0.15 * len(groups)))
        n_val = max(1, round(0.15 * len(groups)))
        if n_test + n_val >= len(groups):
            n_test, n_val = 1, 1
        split_groups["test"].update(groups[:n_test])
        split_groups["val"].update(groups[n_test : n_test + n_val])
        split_groups["train"].update(groups[n_test + n_val :])

    output: dict[str, np.ndarray] = {}
    group_array = np.asarray(group_for_index)
    for split, groups in split_groups.items():
        output[split] = np.flatnonzero(np.isin(group_array, sorted(groups)))
    return output


def load_cache(path: Path, frames: int) -> tuple[np.ndarray, np.ndarray]:
    # The trusted legacy cache stores sequence names as a NumPy object array.
    with np.load(path, allow_pickle=True) as cache:
        raw = np.asarray(cache["sequences"][:, :3], dtype=np.float32)
        names = np.asarray(cache["names"]).astype(str)
    motions = raw.transpose(0, 2, 3, 1)
    if motions.shape[1:] != (frames, 17, 3):
        raise ValueError(f"Expected (N,{frames},17,3), got {motions.shape}")
    if not np.isfinite(motions).all():
        raise ValueError("Motion cache contains non-finite values")
    return motions, names


def make_graph_adjacency(joints: int = 17) -> torch.Tensor:
    adjacency = torch.eye(joints, dtype=torch.float32)
    for left, right in BODY_EDGES:
        adjacency[left, right] = 1.0
        adjacency[right, left] = 1.0
    degree = adjacency.sum(dim=1).clamp_min(1.0)
    inv_sqrt = degree.rsqrt()
    return inv_sqrt[:, None] * adjacency * inv_sqrt[None, :]


def make_mask(
    batch: int,
    frames: int,
    joints: int,
    condition: str,
    generator: torch.Generator,
) -> torch.Tensor:
    mask = torch.ones((batch, frames, joints, 1), dtype=torch.float32)
    for item in range(batch):
        current = condition
        if condition == "train":
            choices = ("random20", "temporal32", "limb48", "mixed")
            current = choices[int(torch.randint(len(choices), (1,), generator=generator))]

        if current.startswith("random") or current == "mixed":
            ratio = float(current.removeprefix("random")) / 100.0 if current.startswith("random") else 0.10
            random_missing = torch.rand((frames, joints), generator=generator) < ratio
            mask[item, :, :, 0][random_missing] = 0.0

        if current.startswith("temporal") or current == "mixed":
            gap = int(current.removeprefix("temporal")) if current.startswith("temporal") else 20
            gap = min(max(gap, 1), frames - 1)
            start = int(torch.randint(0, frames - gap + 1, (1,), generator=generator))
            if current.startswith("temporal"):
                mask[item, start : start + gap, :, 0] = 0.0
            else:
                keep = torch.rand((joints,), generator=generator) > 0.60
                if bool(keep.all()):
                    keep[0] = False
                mask[item, start : start + gap, ~keep, 0] = 0.0

        if current.startswith("limb") or current == "mixed":
            gap = int(current.removeprefix("limb")) if current.startswith("limb") else 48
            gap = min(max(gap, 1), frames - 1)
            start = int(torch.randint(0, frames - gap + 1, (1,), generator=generator))
            limb = LIMB_GROUPS[int(torch.randint(len(LIMB_GROUPS), (1,), generator=generator))]
            mask[item, start : start + gap, list(limb), 0] = 0.0

    return mask


def linear_interpolate(motion: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    batch, frames, joints, coords = motion.shape
    series = motion.permute(0, 2, 3, 1).reshape(batch, joints * coords, frames)
    observed = (
        mask.expand(-1, -1, -1, coords)
        .permute(0, 2, 3, 1)
        .reshape(batch, joints * coords, frames)
        .bool()
    )
    indices = torch.arange(frames, device=motion.device).view(1, 1, frames)
    previous = torch.where(observed, indices, -torch.ones_like(indices)).cummax(dim=-1).values
    following = torch.flip(
        torch.flip(torch.where(observed, indices, torch.full_like(indices, frames)), dims=(-1,))
        .cummin(dim=-1)
        .values,
        dims=(-1,),
    )
    previous_clamped = previous.clamp(0, frames - 1)
    following_clamped = following.clamp(0, frames - 1)
    previous_value = series.gather(-1, previous_clamped)
    following_value = series.gather(-1, following_clamped)
    has_previous = previous >= 0
    has_following = following < frames
    denominator = (following - previous).clamp_min(1)
    weight = (indices - previous).to(series.dtype) / denominator.to(series.dtype)
    interpolated = previous_value + weight * (following_value - previous_value)
    interpolated = torch.where(has_previous & ~has_following, previous_value, interpolated)
    interpolated = torch.where(~has_previous & has_following, following_value, interpolated)
    interpolated = torch.where(has_previous | has_following, interpolated, torch.zeros_like(series))
    output = torch.where(observed, series, interpolated)
    return output.reshape(batch, joints, coords, frames).permute(0, 3, 1, 2)


@torch.no_grad()
def hankel_low_rank(
    motion: torch.Tensor,
    observed_motion: torch.Tensor,
    mask: torch.Tensor,
    window: int,
    rank: int,
) -> torch.Tensor:
    batch, frames, joints, coords = motion.shape
    # CUDA eigendecomposition is not available in FP16, so this small structured
    # projection deliberately stays in FP32 while the neural path uses AMP.
    with torch.autocast(device_type=motion.device.type, enabled=False):
        working = motion.float()
        series = working.permute(0, 2, 3, 1).reshape(batch, joints * coords, frames)
        mean = series.mean(dim=-1, keepdim=True)
        centered = series - mean
        hankel = centered.unfold(-1, window, 1)
        length = hankel.shape[-2]
        covariance = torch.matmul(hankel.transpose(-1, -2), hankel) / max(length, 1)
        _, eigenvectors = torch.linalg.eigh(covariance)
        basis = eigenvectors[..., -min(rank, window) :]
        projected = torch.matmul(torch.matmul(hankel, basis), basis.transpose(-1, -2))
        folded = torch.zeros_like(centered)
        counts = torch.zeros((frames,), dtype=working.dtype, device=working.device)
        for offset in range(window):
            folded[..., offset : offset + length] += projected[..., :, offset]
            counts[offset : offset + length] += 1.0
        reconstructed = folded / counts.clamp_min(1.0) + mean
        reconstructed = reconstructed.reshape(batch, joints, coords, frames).permute(0, 3, 1, 2)
        return torch.where(mask.bool(), observed_motion.float(), reconstructed)


class GraphBlock(nn.Module):
    def __init__(self, width: int, adjacency: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("adjacency", adjacency)
        self.norm = nn.LayerNorm(width)
        self.linear = nn.Linear(width, width)
        self.feed_forward = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width * 2),
            nn.GELU(),
            nn.Linear(width * 2, width),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        mixed = torch.einsum("ij,btjd->btid", self.adjacency, self.norm(features))
        features = features + F.gelu(self.linear(mixed))
        return features + self.feed_forward(features)


class ReliabilityGraphBlock(nn.Module):
    def __init__(self, width: int, adjacency: torch.Tensor, reliability_aware: bool = True) -> None:
        super().__init__()
        self.register_buffer("adjacency", adjacency)
        self.reliability_aware = reliability_aware
        self.norm = nn.LayerNorm(width)
        self.linear = nn.Linear(width, width)
        self.feed_forward = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width * 2),
            nn.GELU(),
            nn.Linear(width * 2, width),
        )

    def forward(self, features: torch.Tensor, reliability: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(features)
        standard_numerator = torch.einsum(
            "ij,btjd->btid", self.adjacency, normalized
        )
        standard_denominator = self.adjacency.sum(dim=1).view(1, 1, -1, 1)
        standard = standard_numerator / standard_denominator.clamp_min(1e-6)
        if self.reliability_aware:
            numerator = torch.einsum(
                "ij,btjd->btid", self.adjacency, normalized * reliability
            )
            denominator = torch.einsum(
                "ij,btjv->btiv", self.adjacency, reliability
            )
            reliable = numerator / denominator.clamp_min(1e-6)
            mixed = torch.where(denominator > 1e-6, reliable, standard)
        else:
            mixed = standard
        features = features + F.gelu(self.linear(mixed))
        return features + self.feed_forward(features)


class TemporalConvNet(nn.Module):
    def __init__(
        self,
        frames: int,
        joints: int = 17,
        width: int = 192,
        dilations: tuple[int, ...] = (1, 2, 4, 8, 16),
    ) -> None:
        super().__init__()
        channels = joints * 4
        self.frames = frames
        self.joints = joints
        self.input = nn.Conv1d(channels, width, kernel_size=1)
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
        features = torch.cat((initial, mask), dim=-1).flatten(2).transpose(1, 2)
        hidden = self.input(features)
        for block in self.blocks:
            hidden = hidden + block(hidden)
        residual = self.output(hidden).transpose(1, 2).reshape_as(initial)
        prediction = initial + residual
        return torch.where(mask.bool(), motion, prediction)


def contiguous_missing_lengths(mask: torch.Tensor) -> torch.Tensor:
    missing = ~mask[..., 0].bool()
    frames = missing.shape[1]
    indices = torch.arange(frames, device=mask.device, dtype=torch.int64).view(
        1, frames, 1
    )
    observed = ~missing
    previous_observed = torch.where(
        observed, indices, torch.full_like(indices, -1)
    ).cummax(dim=1).values
    next_observed = torch.flip(
        torch.flip(
            torch.where(observed, indices, torch.full_like(indices, frames)),
            dims=(1,),
        ).cummin(dim=1).values,
        dims=(1,),
    )
    lengths = next_observed - (previous_observed + 1)
    return torch.where(missing, lengths, torch.zeros_like(lengths))


class GapAdaptiveResidualCompletion(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        zero_through: int = 1,
        full_from: int = 8,
    ) -> None:
        super().__init__()
        if full_from <= zero_through:
            raise ValueError("full_from must be greater than zero_through")
        self.backbone = backbone
        self.zero_through = zero_through
        self.full_from = full_from

    def forward(self, motion: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        initial = linear_interpolate(motion, mask)
        learned = self.backbone(motion, mask)
        run_length = contiguous_missing_lengths(mask).to(learned.dtype)
        weight = (
            (run_length - self.zero_through)
            / (self.full_from - self.zero_through)
        ).clamp(0.0, 1.0)[..., None]
        prediction = initial + weight * (learned - initial)
        return torch.where(mask.bool(), motion, prediction)


class DurationAwareExpertRouter(nn.Module):
    def __init__(
        self,
        short_expert: nn.Module,
        long_expert: nn.Module,
        linear_through: int = 1,
        short_full_from: int = 8,
        long_transition_start: int = 16,
        long_full_from: int = 32,
    ) -> None:
        super().__init__()
        if short_full_from <= linear_through:
            raise ValueError("short_full_from must exceed linear_through")
        if long_full_from <= long_transition_start:
            raise ValueError("long_full_from must exceed long_transition_start")
        self.short_expert = short_expert
        self.long_expert = long_expert
        self.linear_through = linear_through
        self.short_full_from = short_full_from
        self.long_transition_start = long_transition_start
        self.long_full_from = long_full_from

    def forward(self, motion: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        initial = linear_interpolate(motion, mask)
        short_prediction = self.short_expert(motion, mask)
        long_prediction = self.long_expert(motion, mask)
        run_length = contiguous_missing_lengths(mask).to(initial.dtype)
        short_weight = (
            (run_length - self.linear_through)
            / (self.short_full_from - self.linear_through)
        ).clamp(0.0, 1.0)[..., None]
        short_routed = initial + short_weight * (short_prediction - initial)
        long_weight = (
            (run_length - self.long_transition_start)
            / (self.long_full_from - self.long_transition_start)
        ).clamp(0.0, 1.0)[..., None]
        prediction = short_routed + long_weight * (
            long_prediction - short_routed
        )
        return torch.where(mask.bool(), motion, prediction)


class BiGRUCompletion(nn.Module):
    def __init__(self, frames: int, joints: int = 17, width: int = 128) -> None:
        super().__init__()
        self.frames = frames
        self.joints = joints
        self.embedding = nn.Linear(joints * 4, width)
        self.recurrent = nn.GRU(
            width,
            width,
            num_layers=2,
            dropout=0.05,
            bidirectional=True,
            batch_first=True,
        )
        self.output = nn.Linear(width * 2, joints * 3)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, motion: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        initial = linear_interpolate(motion, mask)
        features = self.embedding(torch.cat((initial, mask), dim=-1).flatten(2))
        hidden, _ = self.recurrent(features)
        residual = self.output(hidden).reshape_as(initial)
        return torch.where(mask.bool(), motion, initial + residual)


class TransformerCompletion(nn.Module):
    def __init__(self, frames: int, joints: int = 17, width: int = 128) -> None:
        super().__init__()
        self.frames = frames
        self.joints = joints
        self.embedding = nn.Linear(joints * 4, width)
        self.position = nn.Parameter(torch.zeros(1, frames, width))
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=4,
            dim_feedforward=width * 3,
            dropout=0.05,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(layer, num_layers=3)
        self.output = nn.Linear(width, joints * 3)
        nn.init.trunc_normal_(self.position, std=0.02)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, motion: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        initial = linear_interpolate(motion, mask)
        features = self.embedding(torch.cat((initial, mask), dim=-1).flatten(2)) + self.position
        residual = self.output(self.temporal(features)).reshape_as(initial)
        return torch.where(mask.bool(), motion, initial + residual)


class PatchTransformerCompletion(nn.Module):
    def __init__(self, frames: int, joints: int = 17, width: int = 128) -> None:
        super().__init__()
        self.frames = frames
        self.joints = joints
        self.max_part_joints = max(len(part) for part in BODY_PARTS)
        self.embedding = nn.Linear(self.max_part_joints * 4, width)
        self.time_position = nn.Parameter(torch.zeros(1, frames, 1, width))
        self.part_position = nn.Parameter(torch.zeros(1, 1, len(BODY_PARTS), width))
        spatial_layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=4,
            dim_feedforward=width * 3,
            dropout=0.05,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=4,
            dim_feedforward=width * 3,
            dropout=0.05,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.spatial_encoder = nn.TransformerEncoder(spatial_layer, num_layers=2)
        self.temporal_encoder = nn.TransformerEncoder(temporal_layer, num_layers=3)
        self.output_heads = nn.ModuleList(
            [nn.Linear(width, len(part) * 3) for part in BODY_PARTS]
        )
        nn.init.trunc_normal_(self.time_position, std=0.02)
        nn.init.trunc_normal_(self.part_position, std=0.02)
        for head in self.output_heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, motion: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        initial = linear_interpolate(motion, mask)
        batch, frames, _, _ = initial.shape
        part_tokens: list[torch.Tensor] = []
        combined = torch.cat((initial, mask), dim=-1)
        for part in BODY_PARTS:
            values = combined[:, :, list(part)]
            padded = torch.zeros(
                (batch, frames, self.max_part_joints, 4),
                dtype=values.dtype,
                device=values.device,
            )
            padded[:, :, : len(part)] = values
            part_tokens.append(self.embedding(padded.flatten(2)))
        tokens = torch.stack(part_tokens, dim=2)
        tokens = tokens + self.time_position + self.part_position
        parts = len(BODY_PARTS)
        encoded = self.spatial_encoder(tokens.reshape(batch * frames, parts, -1))
        encoded = encoded.reshape(batch, frames, parts, -1).permute(0, 2, 1, 3)
        encoded = self.temporal_encoder(encoded.reshape(batch * parts, frames, -1))
        encoded = encoded.reshape(batch, parts, frames, -1).permute(0, 2, 1, 3)
        residual = torch.zeros_like(initial)
        for part_index, (part, head) in enumerate(zip(BODY_PARTS, self.output_heads, strict=True)):
            values = head(encoded[:, :, part_index]).reshape(batch, frames, len(part), 3)
            residual[:, :, list(part)] = values.to(residual.dtype)
        return torch.where(mask.bool(), motion, initial + residual)


class DanceFillNet(nn.Module):
    def __init__(
        self,
        frames: int,
        hankel_window: int,
        hankel_rank: int,
        joints: int = 17,
        joint_width: int = 20,
        model_width: int = 192,
        use_graph: bool = True,
        use_hankel: bool = True,
        use_velocity: bool = True,
        dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32),
    ) -> None:
        super().__init__()
        self.frames = frames
        self.joints = joints
        self.hankel_window = hankel_window
        self.hankel_rank = hankel_rank
        self.use_hankel = use_hankel
        self.use_velocity = use_velocity
        adjacency = make_graph_adjacency(joints)
        # Coordinates, Hankel correction, local velocity, and observation mask.
        self.joint_embedding = nn.Linear(10, joint_width)
        self.graph_blocks = nn.ModuleList(
            [GraphBlock(joint_width, adjacency), GraphBlock(joint_width, adjacency)] if use_graph else []
        )
        self.temporal_input = nn.Conv1d(joints * joint_width, model_width, kernel_size=1)
        temporal_blocks: list[nn.Module] = []
        for dilation in dilations:
            temporal_blocks.append(
                nn.Sequential(
                    nn.Conv1d(
                        model_width,
                        model_width * 2,
                        kernel_size=3,
                        padding=dilation,
                        dilation=dilation,
                    ),
                    nn.GLU(dim=1),
                    nn.Conv1d(model_width, model_width, kernel_size=1),
                    nn.GELU(),
                )
            )
        self.temporal_blocks = nn.ModuleList(temporal_blocks)
        self.output = nn.Conv1d(model_width, joints * 3, kernel_size=1)
        # The correction head starts from the strong interpolation solution.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, motion: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        linear = linear_interpolate(motion, mask)
        if self.use_hankel:
            hankel = hankel_low_rank(
                linear,
                observed_motion=motion,
                mask=mask,
                window=self.hankel_window,
                rank=self.hankel_rank,
            )
        else:
            hankel = linear
        velocity = (
            torch.diff(linear, dim=1, prepend=linear[:, :1])
            if self.use_velocity
            else torch.zeros_like(linear)
        )
        joint_input = torch.cat((linear, hankel - linear, velocity, mask), dim=-1)
        joint_features = self.joint_embedding(joint_input)
        for block in self.graph_blocks:
            joint_features = block(joint_features)
        hidden = self.temporal_input(joint_features.flatten(2).transpose(1, 2))
        for block in self.temporal_blocks:
            hidden = hidden + block(hidden)
        residual = self.output(hidden).transpose(1, 2).reshape_as(linear)
        prediction = linear + residual
        return torch.where(mask.bool(), motion, prediction)


class DanceFillGatedNet(nn.Module):
    def __init__(
        self,
        frames: int,
        joints: int = 17,
        joint_width: int = 16,
        model_width: int = 160,
        use_graph: bool = True,
        use_velocity: bool = True,
        learn_gate: bool = True,
        use_severity_prior: bool = True,
        branch_mode: str = "dual",
    ) -> None:
        super().__init__()
        if branch_mode not in {"dual", "short", "long"}:
            raise ValueError(f"Unknown branch mode: {branch_mode}")
        self.frames = frames
        self.joints = joints
        self.use_velocity = use_velocity
        self.learn_gate = learn_gate
        self.use_severity_prior = use_severity_prior
        self.branch_mode = branch_mode
        self.joint_embedding = nn.Linear(7, joint_width)
        adjacency = make_graph_adjacency(joints)
        self.graph_blocks = nn.ModuleList(
            [GraphBlock(joint_width, adjacency), GraphBlock(joint_width, adjacency)]
            if use_graph
            else []
        )
        self.temporal_input = nn.Conv1d(joints * joint_width, model_width, kernel_size=1)
        self.short_blocks = nn.ModuleList(
            [self._temporal_block(model_width, dilation) for dilation in (1, 2, 4, 8)]
        )
        self.long_blocks = nn.ModuleList(
            [self._temporal_block(model_width, dilation) for dilation in (1, 4, 16, 32)]
        )
        self.gate = nn.Sequential(
            nn.Conv1d(model_width + 1, model_width // 2, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(model_width // 2, 1, kernel_size=1),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)
        self.output = nn.Conv1d(model_width, joints * 3, kernel_size=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.last_gate: torch.Tensor | None = None

    @staticmethod
    def _temporal_block(width: int, dilation: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv1d(
                width,
                width * 2,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
            ),
            nn.GLU(dim=1),
            nn.Conv1d(width, width, kernel_size=1),
            nn.GELU(),
        )

    @staticmethod
    def _run_branch(hidden: torch.Tensor, blocks: nn.ModuleList) -> torch.Tensor:
        output = hidden
        for block in blocks:
            output = output + block(output)
        return output

    def forward(self, motion: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        initial = linear_interpolate(motion, mask)
        velocity = (
            torch.diff(initial, dim=1, prepend=initial[:, :1])
            if self.use_velocity
            else torch.zeros_like(initial)
        )
        joint_input = torch.cat((initial, velocity, mask), dim=-1)
        joint_features = self.joint_embedding(joint_input)
        for block in self.graph_blocks:
            joint_features = block(joint_features)
        hidden = self.temporal_input(joint_features.flatten(2).transpose(1, 2))

        if self.branch_mode == "short":
            fused = self._run_branch(hidden, self.short_blocks)
            gate = torch.zeros(
                (len(hidden), 1, hidden.shape[-1]),
                dtype=hidden.dtype,
                device=hidden.device,
            )
        elif self.branch_mode == "long":
            fused = self._run_branch(hidden, self.long_blocks)
            gate = torch.ones(
                (len(hidden), 1, hidden.shape[-1]),
                dtype=hidden.dtype,
                device=hidden.device,
            )
        else:
            short = self._run_branch(hidden, self.short_blocks)
            long = self._run_branch(hidden, self.long_blocks)
            visibility = mask.mean(dim=2).transpose(1, 2)
            severity_prior = 2.0 * (1.0 - visibility) - 1.0
            learned_logits = self.gate(torch.cat((hidden, visibility), dim=1))
            gate_logits = learned_logits if self.learn_gate else torch.zeros_like(learned_logits)
            if self.use_severity_prior:
                gate_logits = gate_logits + severity_prior
            gate = torch.sigmoid(gate_logits)
            fused = (1.0 - gate) * short + gate * long
        self.last_gate = gate.detach()

        residual = self.output(fused).transpose(1, 2).reshape_as(initial)
        prediction = initial + residual
        return torch.where(mask.bool(), motion, prediction)


class DanceFillReliabilityNet(nn.Module):
    def __init__(
        self,
        frames: int,
        joints: int = 17,
        joint_width: int = 16,
        model_width: int = 244,
        use_graph: bool = True,
        reliability_aware: bool = True,
        use_velocity: bool = True,
    ) -> None:
        super().__init__()
        self.frames = frames
        self.joints = joints
        self.use_velocity = use_velocity
        self.joint_embedding = nn.Linear(7, joint_width)
        adjacency = make_graph_adjacency(joints)
        self.graph_blocks = nn.ModuleList(
            [
                ReliabilityGraphBlock(joint_width, adjacency, reliability_aware),
                ReliabilityGraphBlock(joint_width, adjacency, reliability_aware),
            ]
            if use_graph
            else []
        )
        self.temporal_input = nn.Conv1d(joints * joint_width, model_width, kernel_size=1)
        self.temporal_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        model_width,
                        model_width,
                        kernel_size=3,
                        padding=dilation,
                        dilation=dilation,
                    ),
                    nn.GELU(),
                    nn.Conv1d(model_width, model_width, kernel_size=1),
                    nn.GELU(),
                )
                for dilation in (1, 2, 4, 8, 16, 32)
            ]
        )
        self.output = nn.Conv1d(model_width, joints * 3, kernel_size=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, motion: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        initial = linear_interpolate(motion, mask)
        velocity = (
            torch.diff(initial, dim=1, prepend=initial[:, :1])
            if self.use_velocity
            else torch.zeros_like(initial)
        )
        joint_input = torch.cat((initial, velocity, mask), dim=-1)
        joint_features = self.joint_embedding(joint_input)
        for block in self.graph_blocks:
            joint_features = block(joint_features, mask)
        hidden = self.temporal_input(joint_features.flatten(2).transpose(1, 2))
        for block in self.temporal_blocks:
            hidden = hidden + block(hidden)
        residual = self.output(hidden).transpose(1, 2).reshape_as(initial)
        prediction = initial + residual
        return torch.where(mask.bool(), motion, prediction)


def bone_lengths(motion: torch.Tensor) -> torch.Tensor:
    left = torch.as_tensor([edge[0] for edge in BODY_EDGES], device=motion.device)
    right = torch.as_tensor([edge[1] for edge in BODY_EDGES], device=motion.device)
    return torch.linalg.vector_norm(motion[:, :, left] - motion[:, :, right], dim=-1)


@torch.no_grad()
def kinematic_project(
    prediction: torch.Tensor,
    observed_motion: torch.Tensor,
    mask: torch.Tensor,
    iterations: int = 3,
) -> torch.Tensor:
    initial = linear_interpolate(observed_motion, mask)
    left = torch.as_tensor([edge[0] for edge in BODY_EDGES], device=prediction.device)
    right = torch.as_tensor([edge[1] for edge in BODY_EDGES], device=prediction.device)
    observed_bones = torch.linalg.vector_norm(
        observed_motion[:, :, left] - observed_motion[:, :, right], dim=-1
    )
    valid = mask[:, :, left, 0].bool() & mask[:, :, right, 0].bool()
    length_sum = (observed_bones * valid).sum(dim=1)
    valid_count = valid.sum(dim=1)
    fallback = bone_lengths(initial).median(dim=1).values
    target_lengths = torch.where(
        valid_count > 0,
        length_sum / valid_count.clamp_min(1),
        fallback,
    )

    output = prediction.float().clone()
    for _ in range(max(0, iterations)):
        for edge_index, (left_joint, right_joint) in enumerate(BODY_EDGES):
            left_position = output[:, :, left_joint]
            right_position = output[:, :, right_joint]
            direction = left_position - right_position
            current_length = torch.linalg.vector_norm(direction, dim=-1, keepdim=True)
            unit = direction / current_length.clamp_min(1e-6)
            correction = (
                target_lengths[:, edge_index, None, None] - current_length
            ) * unit
            left_missing = ~mask[:, :, left_joint].bool()
            right_missing = ~mask[:, :, right_joint].bool()
            both_missing = left_missing & right_missing
            only_left = left_missing & ~right_missing
            only_right = right_missing & ~left_missing
            output[:, :, left_joint] = torch.where(
                both_missing,
                left_position + 0.5 * correction,
                torch.where(only_left, left_position + correction, left_position),
            )
            output[:, :, right_joint] = torch.where(
                both_missing,
                right_position - 0.5 * correction,
                torch.where(only_right, right_position - correction, right_position),
            )
        output = torch.where(mask.bool(), observed_motion.float(), output)
    return output.to(prediction.dtype)


@torch.no_grad()
def adaptive_kinematic_project(
    prediction: torch.Tensor,
    observed_motion: torch.Tensor,
    mask: torch.Tensor,
    iterations: int = 2,
    full_weight_until: int = 48,
    zero_weight_from: int = 64,
) -> torch.Tensor:
    projected = kinematic_project(
        prediction,
        observed_motion,
        mask,
        iterations=iterations,
    )
    missing = ~mask[..., 0].bool()
    current = torch.zeros(
        (len(mask), mask.shape[2]), dtype=torch.int64, device=mask.device
    )
    longest = torch.zeros_like(current)
    for frame in range(mask.shape[1]):
        current = torch.where(missing[:, frame], current + 1, torch.zeros_like(current))
        longest = torch.maximum(longest, current)
    denominator = max(zero_weight_from - full_weight_until, 1)
    joint_weight = (
        (zero_weight_from - longest).to(prediction.dtype) / denominator
    ).clamp(0.0, 1.0)
    all_missing_frame = missing.all(dim=2, keepdim=True)
    weight = joint_weight[:, None, :, None] * (~all_missing_frame)[..., None]
    blended = prediction + weight * (projected - prediction)
    return torch.where(mask.bool(), observed_motion, blended)


class AdaptiveKinematicCompletion(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        iterations: int = 2,
        full_weight_until: int = 48,
        zero_weight_from: int = 64,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.iterations = iterations
        self.full_weight_until = full_weight_until
        self.zero_weight_from = zero_weight_from

    def forward(self, motion: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        prediction = self.backbone(motion, mask)
        return adaptive_kinematic_project(
            prediction,
            motion,
            mask,
            iterations=self.iterations,
            full_weight_until=self.full_weight_until,
            zero_weight_from=self.zero_weight_from,
        )


class FixedKinematicCompletion(nn.Module):
    def __init__(self, backbone: nn.Module, iterations: int = 2) -> None:
        super().__init__()
        self.backbone = backbone
        self.iterations = iterations

    def forward(self, motion: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        prediction = self.backbone(motion, mask)
        return kinematic_project(
            prediction,
            motion,
            mask,
            iterations=self.iterations,
        )


def completion_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    config: PilotConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    missing = 1.0 - mask
    position_distance = torch.linalg.vector_norm(prediction - target, dim=-1)
    position = (position_distance * missing[..., 0]).sum() / missing[..., 0].sum().clamp_min(1.0)

    pred_velocity = prediction[:, 1:] - prediction[:, :-1]
    true_velocity = target[:, 1:] - target[:, :-1]
    transition = torch.maximum(missing[:, 1:], missing[:, :-1])
    velocity_distance = torch.linalg.vector_norm(pred_velocity - true_velocity, dim=-1)
    velocity = (velocity_distance * transition[..., 0]).sum() / transition[..., 0].sum().clamp_min(1.0)

    bone = F.smooth_l1_loss(bone_lengths(prediction), bone_lengths(target))
    total = (
        config.position_weight * position
        + config.velocity_weight * velocity
        + config.bone_weight * bone
    )
    parts = {"position": float(position.detach()), "velocity": float(velocity.detach()), "bone": float(bone.detach())}
    return total, parts


def train_model(
    name: str,
    model: nn.Module,
    train_loader: DataLoader[torch.Tensor],
    val_loader: DataLoader[torch.Tensor],
    config: PilotConfig,
    device: torch.device,
    out_dir: Path,
) -> nn.Module:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    generator = torch.Generator().manual_seed(config.seed + 17)
    best_mpjpe = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float | int]] = []
    stale_epochs = 0

    for epoch in range(1, config.epochs + 1):
        if hasattr(train_loader.dataset, "set_epoch"):
            train_loader.dataset.set_epoch(epoch)
        model.train()
        total_loss = 0.0
        samples = 0
        started = time.perf_counter()
        for target in train_loader:
            target = target.to(device, non_blocking=True)
            mask = make_mask(
                len(target), target.shape[1], target.shape[2], "train", generator
            ).to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                prediction = model(target, mask)
                loss, _ = completion_loss(prediction, target, mask, config)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach()) * len(target)
            samples += len(target)

        validation = evaluate_method(
            model, val_loader, "mixed", config.seed + 1000, device, warmup=False
        )
        record = {
            "epoch": epoch,
            "train_loss": total_loss / max(samples, 1),
            "val_mpjpe": validation["mpjpe"],
            "seconds": time.perf_counter() - started,
        }
        history.append(record)
        print(
            f"[{name}] epoch={epoch:02d} loss={record['train_loss']:.6f} "
            f"val_mpjpe={record['val_mpjpe']:.6f} seconds={record['seconds']:.1f}",
            flush=True,
        )
        if validation["mpjpe"] < best_mpjpe - 1e-6:
            best_mpjpe = validation["mpjpe"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= 5:
            print(f"[{name}] early stopping at epoch {epoch}", flush=True)
            break

    if best_state is None:
        raise RuntimeError(f"No checkpoint produced for {name}")
    model.load_state_dict(best_state)
    torch.save({"model": best_state, "config": asdict(config)}, out_dir / f"{name}_best.pt")
    with (out_dir / f"{name}_history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    return model


@torch.no_grad()
def evaluate_method(
    method: nn.Module | str,
    loader: DataLoader[torch.Tensor],
    condition: str,
    seed: int,
    device: torch.device,
    warmup: bool = True,
    hankel_window: int = 12,
    hankel_rank: int = 4,
) -> dict[str, float]:
    if isinstance(method, nn.Module):
        method.eval()
    generator = torch.Generator().manual_seed(seed)
    totals = {"distance": 0.0, "points": 0.0, "velocity": 0.0, "transitions": 0.0, "bone": 0.0, "bones": 0.0}
    elapsed = 0.0
    sequences = 0
    for batch_index, target in enumerate(loader):
        target = target.to(device, non_blocking=True)
        mask = make_mask(len(target), target.shape[1], target.shape[2], condition, generator).to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        initial = linear_interpolate(target, mask)
        if method == "linear":
            prediction = initial
        elif method == "hankel":
            prediction = hankel_low_rank(initial, target, mask, hankel_window, hankel_rank)
        elif isinstance(method, nn.Module):
            prediction = method(target, mask)
        else:
            raise ValueError(f"Unknown method: {method}")
        if device.type == "cuda":
            torch.cuda.synchronize()
        duration = time.perf_counter() - started
        if not warmup or batch_index > 0:
            elapsed += duration
            sequences += len(target)

        missing = (1.0 - mask[..., 0]).bool()
        distance = torch.linalg.vector_norm(prediction - target, dim=-1)
        totals["distance"] += float(distance[missing].sum())
        totals["points"] += float(missing.sum())

        pred_velocity = prediction[:, 1:] - prediction[:, :-1]
        true_velocity = target[:, 1:] - target[:, :-1]
        velocity_distance = torch.linalg.vector_norm(pred_velocity - true_velocity, dim=-1)
        transition = missing[:, 1:] | missing[:, :-1]
        totals["velocity"] += float(velocity_distance[transition].sum())
        totals["transitions"] += float(transition.sum())

        pred_bone = bone_lengths(prediction)
        true_bone = bone_lengths(target)
        left = torch.as_tensor([edge[0] for edge in BODY_EDGES], device=device)
        right = torch.as_tensor([edge[1] for edge in BODY_EDGES], device=device)
        bone_missing = missing[:, :, left] | missing[:, :, right]
        bone_relative = (pred_bone - true_bone).abs() / true_bone.clamp_min(1e-3)
        totals["bone"] += float(bone_relative[bone_missing].sum())
        totals["bones"] += float(bone_missing.sum())

    return {
        "mpjpe": totals["distance"] / max(totals["points"], 1.0),
        "velocity_error": totals["velocity"] / max(totals["transitions"], 1.0),
        "bone_relative_error": totals["bone"] / max(totals["bones"], 1.0),
        "milliseconds_per_sequence": 1000.0 * elapsed / max(sequences, 1),
    }


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def write_results(rows: list[dict[str, object]], out_dir: Path) -> None:
    fields = list(rows[0])
    with (out_dir / "pilot_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with (out_dir / "pilot_results.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("data/sequence_cache_128.npz"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("runs/pilot_v1"),
    )
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    config = PilotConfig(
        epochs=3 if args.quick else args.epochs,
        batch_size=16 if args.quick else args.batch_size,
    )
    seed_everything(config.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    motions, names = load_cache(args.cache, config.frames)
    splits = grouped_split(names, config.seed)
    if args.quick:
        splits = {
            "train": splits["train"][:256],
            "val": splits["val"][:96],
            "test": splits["test"][:128],
        }
    split_manifest = {
        split: {
            "count": int(len(indices)),
            "groups": sorted({choreography_group(names[index])[1] for index in indices}),
        }
        for split, indices in splits.items()
    }
    with (args.out_dir / "split_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(split_manifest, handle, indent=2)

    loader_kwargs = {
        "batch_size": config.batch_size,
        "num_workers": 2,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(
        MotionDataset(motions, splits["train"]), shuffle=True, drop_last=False, **loader_kwargs
    )
    val_loader = DataLoader(
        MotionDataset(motions, splits["val"]), shuffle=False, drop_last=False, **loader_kwargs
    )
    test_loader = DataLoader(
        MotionDataset(motions, splits["test"]), shuffle=False, drop_last=False, **loader_kwargs
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} splits={ {key: len(value) for key, value in splits.items()} }", flush=True)

    tcn = TemporalConvNet(config.frames).to(device)
    dancefill = DanceFillNet(
        config.frames, config.hankel_window, config.hankel_rank
    ).to(device)
    print(f"tcn_params={parameter_count(tcn)} dancefill_params={parameter_count(dancefill)}", flush=True)
    tcn = train_model("tcn", tcn, train_loader, val_loader, config, device, args.out_dir)
    dancefill = train_model(
        "dancefill", dancefill, train_loader, val_loader, config, device, args.out_dir
    )

    methods: list[tuple[str, nn.Module | str, int]] = [
        ("linear", "linear", 0),
        ("hankel", "hankel", 0),
        ("tcn", tcn, parameter_count(tcn)),
        ("dancefill", dancefill, parameter_count(dancefill)),
    ]
    conditions = ("random20", "temporal32", "limb48", "mixed")
    rows: list[dict[str, object]] = []
    for method_name, method, parameters in methods:
        for condition_index, condition in enumerate(conditions):
            metrics = evaluate_method(
                method,
                test_loader,
                condition,
                config.seed + 10_000 + condition_index,
                device,
                hankel_window=config.hankel_window,
                hankel_rank=config.hankel_rank,
            )
            row: dict[str, object] = {
                "method": method_name,
                "condition": condition,
                "parameters": parameters,
                **metrics,
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
    write_results(rows, args.out_dir)

    linear = {row["condition"]: float(row["mpjpe"]) for row in rows if row["method"] == "linear"}
    proposed = {row["condition"]: float(row["mpjpe"]) for row in rows if row["method"] == "dancefill"}
    gains = {
        condition: 100.0 * (linear[condition] - proposed[condition]) / linear[condition]
        for condition in conditions
    }
    gate_passed = gains["limb48"] >= 8.0 and gains["mixed"] >= 8.0
    summary = {
        "config": asdict(config),
        "splits": {key: len(value) for key, value in splits.items()},
        "mpjpe_gain_percent_vs_linear": gains,
        "evidence_gate_passed": gate_passed,
    }
    with (args.out_dir / "pilot_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
