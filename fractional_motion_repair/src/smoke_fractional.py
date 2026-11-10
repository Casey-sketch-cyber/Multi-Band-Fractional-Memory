from __future__ import annotations

import torch

from fractional_models import MemoryTemporalConvNet, validate_fractional_definition
from pilot import GapAdaptiveResidualCompletion, TemporalConvNet, parameter_count


def main() -> None:
    validate_fractional_definition()
    torch.manual_seed(7)
    motion = torch.randn(2, 128, 17, 3)
    mask = torch.ones(2, 128, 17, 1)
    mask[:, 32:80, 5:11] = 0.0

    base = TemporalConvNet(
        128,
        width=248,
        dilations=(1, 2, 4, 8, 16, 32),
    )
    fractional = MemoryTemporalConvNet(128, mode="fractional")
    integer = MemoryTemporalConvNet(128, mode="integer")
    missing_fractional, unexpected_fractional = fractional.load_state_dict(
        base.state_dict(), strict=False
    )
    missing_integer, unexpected_integer = integer.load_state_dict(
        base.state_dict(), strict=False
    )
    if unexpected_fractional or unexpected_integer:
        raise RuntimeError("Unexpected base checkpoint keys")
    if not all(key.startswith("memory.") for key in missing_fractional):
        raise RuntimeError("Fractional checkpoint mismatch outside memory adapter")
    if not all(key.startswith("memory.") for key in missing_integer):
        raise RuntimeError("Integer checkpoint mismatch outside memory adapter")

    base_prediction = base(motion, mask)
    fractional_prediction = fractional(motion, mask)
    integer_prediction = integer(motion, mask)
    if not torch.allclose(base_prediction, fractional_prediction, atol=1e-6):
        raise RuntimeError("Fractional adapter must begin as an identity update")
    if not torch.allclose(base_prediction, integer_prediction, atol=1e-6):
        raise RuntimeError("Integer adapter must begin as an identity update")
    if not torch.equal(fractional_prediction[mask.expand_as(motion).bool()], motion[mask.expand_as(motion).bool()]):
        raise RuntimeError("Observed coordinates changed")

    with torch.no_grad():
        fractional.memory.projection.weight.normal_(mean=0.0, std=0.05)
    hidden_target_changed = motion.clone()
    hidden_target_changed[~mask.expand_as(motion).bool()] += 100.0
    original_hidden_prediction = fractional(motion, mask)
    changed_hidden_prediction = fractional(hidden_target_changed, mask)
    if not torch.allclose(
        original_hidden_prediction,
        changed_hidden_prediction,
        atol=1e-6,
    ):
        raise RuntimeError("Masked ground truth leaked into the prediction")

    wrapped_fractional = GapAdaptiveResidualCompletion(fractional)
    loss = wrapped_fractional(motion, mask).square().mean()
    loss.backward()
    if fractional.memory.projection.weight.grad is None:
        raise RuntimeError("Memory projection received no gradient")
    if not torch.isfinite(fractional.memory.projection.weight.grad).all():
        raise RuntimeError("Non-finite memory gradient")

    output = {
        "base_parameters": parameter_count(base),
        "fractional_parameters": parameter_count(fractional),
        "integer_parameters": parameter_count(integer),
        "missing_checkpoint_keys": missing_fractional,
        "initial_orders": fractional.memory.fractional_orders().mean(dim=0).tolist(),
        "shape": list(fractional_prediction.shape),
        "observed_projection_exact": True,
        "masked_ground_truth_invariant": True,
    }
    print(output)


if __name__ == "__main__":
    main()
