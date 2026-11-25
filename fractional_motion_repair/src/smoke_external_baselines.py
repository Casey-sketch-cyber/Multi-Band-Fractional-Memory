"""Run observation, interpolation, and loss checks on synthetic poses."""

import torch
from torch import nn

from external_baselines import SaitsCompletion, pchip_interpolate


def check_pchip() -> None:
    frames = 8
    time = torch.arange(frames, dtype=torch.float32)
    motion = (time[None, :, None, None] ** 2).expand(2, frames, 5, 3).clone()
    motion[1] += 100
    motion[..., 1] *= 2
    mask = torch.zeros((2, frames, 5, 1), dtype=torch.bool)
    mask[:, [1, 3, 5], 0] = True
    mask[:, [1, 3, 5], 4] = True
    mask[:, 3, 1] = True
    mask[:, :, 3] = True
    motion[:, :, 4] *= -1

    completed = pchip_interpolate(motion, mask)
    hidden_nan = pchip_interpolate(torch.where(mask, motion, torch.nan), mask)
    assert completed.shape == motion.shape and completed.dtype == motion.dtype
    torch.testing.assert_close(completed, hidden_nan, rtol=0, atol=0)
    torch.testing.assert_close(completed[mask.expand_as(motion)], motion[mask.expand_as(motion)])
    torch.testing.assert_close(completed[0, 2, 0, 0], torch.tensor(25 / 6))
    torch.testing.assert_close(completed[0, 4, 0, 0], torch.tensor(95 / 6))
    torch.testing.assert_close(completed[:, 0, 0], motion[:, 1, 0])
    torch.testing.assert_close(completed[:, -1, 0], motion[:, 5, 0])
    torch.testing.assert_close(completed[:, :, 1], motion[:, 3:4, 1].expand(-1, frames, -1))
    assert not completed[:, :, 2].count_nonzero()
    torch.testing.assert_close(completed[:, :, 4], -completed[:, :, 0])


class ConstantSaits(nn.Module):
    def forward(self, inputs, missing_mask, attention_mask):
        assert torch.equal(inputs, missing_mask * 2)
        assert not attention_mask[0].diagonal().any()
        assert attention_mask.sum().item() == inputs.shape[1] * (inputs.shape[1] - 1)
        return (torch.ones_like(inputs), 4 * torch.ones_like(inputs), 5 * torch.ones_like(inputs))


def check_saits() -> None:
    torch.manual_seed(20261201)
    motion = torch.full((2, 8, 17, 3), 2.0)
    mask = torch.ones((2, 8, 17, 1), dtype=torch.bool)
    mask[:, 1:6, 1:3] = False
    mask[1] = False
    model = SaitsCompletion(frames=8, features=51)
    model.eval()
    with torch.no_grad():
        completed = model(motion, mask)
        hidden_nan = model(torch.where(mask, motion, torch.nan), mask)
        assert completed.shape == motion.shape and completed.dtype == motion.dtype
        assert torch.isfinite(completed).all()
        torch.testing.assert_close(completed, hidden_nan, rtol=0, atol=0)
        torch.testing.assert_close(completed[mask.expand_as(motion)], motion[mask.expand_as(motion)])

    model.train()
    loss = model.training_loss(motion, mask)
    assert torch.isfinite(loss)
    loss.backward()
    gradients = [param.grad for param in model.parameters() if param.grad is not None]
    assert all(torch.isfinite(grad).all() for grad in gradients)
    assert any(grad.count_nonzero() for grad in gradients)

    model.backbone = ConstantSaits()
    targets = torch.where(mask, motion, 9)
    torch.testing.assert_close(model.training_loss(targets, mask), torch.tensor(6.0))
    torch.testing.assert_close(
        model.training_loss(motion, torch.ones_like(mask)), torch.tensor(2.0),
    )
    torch.testing.assert_close(
        model.training_loss(targets, torch.zeros_like(mask)),
        (targets - 5).abs().mean(),
    )


if __name__ == "__main__":
    import sys

    torch.set_num_threads(1)
    check_pchip()
    if "--pchip-only" not in sys.argv:
        check_saits()
    print("External baseline checks passed.")
