from __future__ import annotations

import torch

from fractional_models import (
    MemoryTemporalConvNet,
    TemporalMemoryAdapter,
    gl_integral_coefficients,
    validate_fractional_definition,
)
from fractional_screen import KERNEL_VARIANTS, build_long_expert
from pilot import GapAdaptiveResidualCompletion, TemporalConvNet, linear_interpolate, parameter_count


@torch.no_grad()
def check_instant_and_uniform_limits() -> None:
    visibility = torch.ones(1, 6, 32)
    visibility[:, :3, 8:24] = 0
    signal = torch.randn_like(visibility)
    signal[:, 3] = torch.arange(32).square() * 0.01
    instant = TemporalMemoryAdapter(6, mode="instant")
    instant.projection.weight[0, 0, 0] = 1
    torch.testing.assert_close(instant(signal, visibility), signal, rtol=0, atol=0)
    instant.projection.weight[0, 3, 0] = 1
    correction = instant(signal, visibility) - signal
    if not correction[0, 0, 8:24].abs().max() > 0:
        raise RuntimeError("K0 must transmit the visible joint's velocity")
    if correction[visibility.bool()].any():
        raise RuntimeError("K0 changed an observed coordinate")

    uniform = TemporalMemoryAdapter(6, mode="integer", finite_windows=(97, 97, 97))
    tempered = TemporalMemoryAdapter(6, mode="tempered")
    uniform.projection.weight.normal_(std=0.05)
    uniform.mix_logits.normal_()
    uniform.gate_logits.normal_()
    tempered.projection.weight.copy_(uniform.projection.weight)
    tempered.mix_logits.copy_(uniform.mix_logits)
    tempered.gate_logits.copy_(uniform.gate_logits)
    tempered.raw_beta.zero_()
    tempered.raw_decay.zero_()
    torch.testing.assert_close(uniform(signal, visibility), tempered(signal, visibility), rtol=0, atol=0)


def check_revision_kernels(base: TemporalConvNet, motion: torch.Tensor, mask: torch.Tensor) -> dict[str, int]:
    check_instant_and_uniform_limits()
    observed = mask.expand_as(motion).bool()
    changed = torch.where(observed, motion, motion + 100)
    lags = torch.arange(97, dtype=torch.float64)
    fixed_orders = lags.new_tensor([0.225, 0.5, 0.775])
    fixed_mean_lags = (gl_integral_coefficients(fixed_orders, 97) * lags[:, None]).sum(dim=0)
    parameter_counts = {}

    for name in KERNEL_VARIANTS:
        expert, fractional_weight = build_long_expert(name)
        if fractional_weight != 0.0:
            raise RuntimeError(f"{name} must use the base loss")
        backbone = expert.backbone
        missing, unexpected = backbone.load_state_dict(base.state_dict(), strict=False)
        if unexpected or not all(key.startswith("memory.") for key in missing):
            raise RuntimeError(f"{name} cannot initialize from the base expert")
        adapter = backbone.memory
        parameter_counts[name] = parameter_count(expert)
        with torch.no_grad():
            if name in {"K1", "K2"}:
                windows = (9, 33, 97) if name == "K1" else (51, 65, 97)
                if adapter.finite_windows != windows:
                    raise RuntimeError(f"{name} finite windows differ from the protocol")
            if name == "K3":
                orders = adapter.fractional_orders()[0].double()
                torch.testing.assert_close(orders, fixed_orders)
                if adapter.raw_orders.requires_grad:
                    raise RuntimeError("K3 orders must remain fixed")
            if name in {"K4", "K5", "K6", "K7"}:
                coefficients = adapter.shared_kernel_coefficients().double()
                expected_bands = 1 if name == "K7" else 3
                if coefficients.shape != (expected_bands, 97) or not (coefficients >= 0).all():
                    raise RuntimeError(f"Invalid {name} kernel weights")
                torch.testing.assert_close(coefficients.sum(dim=-1), lags.new_ones(expected_bands))
                if name == "K4":
                    torch.testing.assert_close(
                        (coefficients * lags).sum(dim=-1), fixed_mean_lags, rtol=0, atol=1e-10
                    )
                if name == "K6":
                    expected = gl_integral_coefficients(fixed_orders, 97).transpose(0, 1)
                    torch.testing.assert_close(coefficients, expected)
            adapter.projection.weight.normal_(std=0.05)
            backbone.output.weight.normal_(std=0.01)

        prediction = expert(motion, mask)
        with torch.no_grad():
            torch.testing.assert_close(prediction, expert(changed, mask), rtol=0, atol=0)
            torch.testing.assert_close(prediction[observed], motion[observed], rtol=0, atol=0)
        prediction.square().mean().backward()
        learned = [adapter.projection.weight]
        if name in {"K5", "K7"}:
            learned.extend((adapter.raw_beta, adapter.raw_decay))
        if name == "K6":
            learned.append(adapter.kernel_logits)
        if any(
            parameter.grad is None
            or not torch.isfinite(parameter.grad).all()
            or not parameter.grad.any()
            for parameter in learned
        ):
            raise RuntimeError(f"Missing, zero or non-finite {name} kernel gradient")
    return parameter_counts


def check_output_projection() -> None:
    frames, joints = 16, 5
    motion = torch.randn(1, frames, joints, 3, dtype=torch.float64)
    mask = torch.ones_like(motion[..., :1])
    bounded_gaps = ((0, 2, 7), (0, 9, 13), (3, 4, 5))
    for joint, start, stop in (*bounded_gaps, (1, 0, 4), (1, 11, frames), (2, 0, frames)):
        mask[:, start:stop, joint] = 0

    model = MemoryTemporalConvNet(
        frames, mode="tempered", joints=joints, width=12, dilations=(1, 2), output_projection=True
    ).double()
    with torch.no_grad():
        model.output.weight.normal_(std=0.1)
        model.output.bias.fill_(0.4)

    outputs = []
    hook = model.output.register_forward_hook(lambda _layer, _args, output: outputs.append(output.detach()))
    prediction = model(motion, mask)
    hook.remove()
    raw_residual = outputs[0].transpose(1, 2).reshape_as(motion)

    # Compare to the explicit endpoint formula for each bounded gap.
    initial = linear_interpolate(motion, mask)
    expected = initial.clone()
    for joint, start, stop in bounded_gaps:
        weight = torch.arange(1, stop - start + 1, dtype=motion.dtype) / (stop - start + 1)
        left = raw_residual[:, start - 1, joint].unsqueeze(1)
        right = raw_residual[:, stop, joint].unsqueeze(1)
        trend = left + weight[None, :, None] * (right - left)
        expected[:, start:stop, joint] += raw_residual[:, start:stop, joint] - trend
    torch.testing.assert_close(prediction, expected, rtol=0, atol=1e-12)

    observed = mask.expand_as(motion).bool()
    torch.testing.assert_close(prediction[observed], motion[observed], rtol=0, atol=0)
    changed = torch.where(observed, motion, motion + 100)
    with torch.no_grad():
        torch.testing.assert_close(prediction, model(changed, mask), rtol=0, atol=0)
    prediction[~observed].square().mean().backward()
    gradient = model.output.weight.grad
    if gradient is None or not torch.isfinite(gradient).all() or not gradient.any():
        raise RuntimeError("B3 projection did not pass a finite, nonzero gradient")


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

    kernel_parameters = check_revision_kernels(base, motion, mask)
    check_output_projection()
    output = {
        "base_parameters": parameter_count(base),
        "fractional_parameters": parameter_count(fractional),
        "integer_parameters": parameter_count(integer),
        "missing_checkpoint_keys": missing_fractional,
        "initial_orders": fractional.memory.fractional_orders().mean(dim=0).tolist(),
        "shape": list(fractional_prediction.shape),
        "observed_projection_exact": True,
        "masked_ground_truth_invariant": True,
        "revision_kernel_parameters": kernel_parameters,
        "output_endpoint_projection_checked": True,
    }
    print(output)


if __name__ == "__main__":
    main()
