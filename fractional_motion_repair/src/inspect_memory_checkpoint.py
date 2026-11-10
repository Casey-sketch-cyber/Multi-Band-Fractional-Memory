from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def summarize(path: Path) -> dict[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload["model"]
    prefix = "backbone.memory."
    mixing = torch.softmax(state[f"{prefix}mix_logits"], dim=1)
    entropy = -(mixing * mixing.clamp_min(1e-12).log()).sum(dim=1).mean()
    return {
        "checkpoint": str(path),
        "projection_norm": float(state[f"{prefix}projection.weight"].norm()),
        "gate_mean": float(torch.sigmoid(state[f"{prefix}gate_logits"]).mean()),
        "mix_entropy": float(entropy),
        "raw_orders": state[f"{prefix}raw_orders"].tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+", type=Path)
    args = parser.parse_args()
    print(json.dumps([summarize(path) for path in args.checkpoints], indent=2))


if __name__ == "__main__":
    main()
