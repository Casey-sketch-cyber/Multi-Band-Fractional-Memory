from __future__ import annotations

import pickle
from collections.abc import Iterable
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import numpy as np
import torch


COCO_JOINTS = 17


def load_aist_smpl_scales(motion_zip: Path, sequences: Iterable[str]) -> dict[str, float]:
    """Read the available per-sequence scales from the trusted AIST++ motion release."""
    requested = set(sequences)
    scales: dict[str, float] = {}
    with ZipFile(motion_zip) as archive:
        for member in archive.infolist():
            path = PurePosixPath(member.filename)
            if path.parent.name != "motions" or path.suffix != ".pkl":
                continue
            name = path.stem
            if name not in requested:
                continue
            if name in scales:
                raise ValueError(f"Multiple SMPL annotations for {name}")
            with archive.open(member) as handle:
                payload = pickle.load(handle)
            scale = float(np.asarray(payload["smpl_scaling"]).item())
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError(f"Invalid SMPL scaling for {name}: {scale}")
            scales[name] = scale
    return scales


def window_diagnostic_totals(
    distance: torch.Tensor,
    velocity_residual: torch.Tensor,
    missing: torch.Tensor,
) -> np.ndarray:
    """Return per-window sums and support for endpoints, acceleration, and each joint."""
    before, center, after = missing[:, :-2], missing[:, 1:-1], missing[:, 2:]
    endpoint = ~center & (before ^ after)
    acceleration = before | center | after
    acceleration_error = torch.linalg.vector_norm(velocity_residual.diff(dim=1), dim=-1)

    sums = torch.cat(
        (
            acceleration_error.masked_fill(~endpoint, 0).sum(dim=(1, 2)).unsqueeze(1),
            acceleration_error.masked_fill(~acceleration, 0).sum(dim=(1, 2)).unsqueeze(1),
            distance.masked_fill(~missing, 0).sum(dim=1),
        ),
        dim=1,
    )
    counts = torch.cat(
        (
            endpoint.sum(dim=(1, 2)).unsqueeze(1),
            acceleration.sum(dim=(1, 2)).unsqueeze(1),
            missing.sum(dim=1),
        ),
        dim=1,
    )
    return torch.stack((sums, counts), dim=-1).cpu().numpy()


def diagnostic_fields(sums_and_counts: np.ndarray) -> dict[str, float | int | None]:
    endpoint, acceleration, *joints = sums_and_counts
    fields = {
        "endpoint_velocity_jump_error": mean_or_none(*endpoint),
        "endpoint_velocity_jump_points": int(endpoint[1]),
        "acceleration_error": mean_or_none(*acceleration),
        "acceleration_points": int(acceleration[1]),
    }
    fields.update(
        {
            f"mpjpe_joint_{index:02d}": mean_or_none(total, count)
            for index, (total, count) in enumerate(joints)
        }
    )
    return fields


def mean_or_none(total: float, count: float) -> float | None:
    return float(total) / float(count) if count else None
