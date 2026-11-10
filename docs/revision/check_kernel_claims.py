"""核对修订计划中的解析结论（纯 NumPy，无需数据、GPU 或 PyTorch）。

用法（仓库根目录）::

    python docs/revision/check_kernel_claims.py

脚本按 fractional_motion_repair/src/fractional_models.py 的定义重写了三个算子：
GL 分数阶积分系数、掩码归一化双向上下文、端点约束桥，然后逐条核对
docs/revision/MATHEMATICS_REVISION_PLAN.md 第 4 节的命题 P1–P3。
任何一条不成立都会抛出 AssertionError。

注意：这里只验证算子的结构性质（与信号无关），不代表任何真实数据上的精度结论。
"""
from __future__ import annotations

from math import exp, lgamma

import numpy as np

K = 97          # 论文使用的截断长度
FRAMES = 128    # 论文使用的窗口长度


def gl_raw(beta: float, size: int, lam: float = 0.0) -> np.ndarray:
    """未归一化的（回火）GL 积分系数：a_k(beta) * exp(-lam * k)。"""
    weights = [1.0]
    for k in range(1, size):
        weights.append(weights[-1] * (beta + k - 1) / k)
    return np.asarray(weights) * np.exp(-lam * np.arange(size))


def masked_bidirectional(signal: np.ndarray, visible: np.ndarray, coef: np.ndarray) -> np.ndarray:
    """单通道版 masked_bidirectional_context；分母为零时返回原信号。"""
    frames, taps = len(signal), len(coef)
    numerator = np.zeros(frames)
    denominator = np.zeros(frames)
    for t in range(frames):
        for k in range(taps):
            for s in (t - k, t + k):
                if 0 <= s < frames:
                    numerator[t] += coef[k] * visible[s] * signal[s]
                    denominator[t] += coef[k] * visible[s]
    estimate = numerator / np.maximum(denominator, 1e-6)
    return np.where(denominator > 1e-6, estimate, signal)


def endpoint_bridge(residual: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """单通道版 endpoint_constrained_bridge。"""
    frames = len(residual)
    cumulative = np.cumsum(residual)
    correction = np.zeros(frames)
    for t in range(frames):
        if observed[t]:
            continue
        a = t
        while a >= 0 and not observed[a]:
            a -= 1
        b = t
        while b < frames and not observed[b]:
            b += 1
        if a < 0 or b >= frames:
            continue
        correction[t] = (cumulative[t] - cumulative[a]) - (t - a) / (b - a) * (
            cumulative[b] - cumulative[a]
        )
    return correction


def check_p1_closed_forms() -> None:
    print("P1  核的闭式性质")
    for beta in (0.225, 0.5, 0.775):
        raw = gl_raw(beta, K)
        total = raw.sum()
        closed_total = exp(lgamma(K + beta) - lgamma(K) - lgamma(beta + 1))
        mean_lag = float((np.arange(K) * raw).sum() / total)
        closed_lag = beta * (K - 1) / (beta + 1)
        assert abs(total - closed_total) < 1e-8 * closed_total
        assert abs(mean_lag - closed_lag) < 1e-8
        # 渐近幂律：a_k * Gamma(beta) / k^(beta-1) -> 1
        ratio = raw[-1] * exp(lgamma(beta)) / (K - 1) ** (beta - 1)
        assert abs(ratio - 1.0) < 0.01
        print(
            f"    beta={beta:.3f}: 截断和={total:.4f}  平均滞后={mean_lag:.3f}"
            f"  首尾权重比={raw[0] / raw[-1]:.1f}"
        )
    # beta = 1 退化为均匀窗
    assert np.allclose(gl_raw(1.0, K), 1.0)
    print("    beta=1, lambda=0: 权重全为 1（即 97 抽头均匀窗），平均滞后=48")
    # 回火：归一化权重是负二项分布
    for beta, lam in ((0.3, 0.05), (0.7, 0.2), (1.0, 0.1)):
        raw = gl_raw(beta, 6000, lam)
        z = exp(-lam)
        assert abs(raw.sum() - (1 - z) ** (-beta)) < 1e-9
        mean_lag = float((np.arange(len(raw)) * raw).sum() / raw.sum())
        assert abs(mean_lag - beta / (exp(lam) - 1)) < 1e-8
        variance = float((np.arange(len(raw)) ** 2 * raw).sum() / raw.sum() - mean_lag**2)
        assert abs(variance - beta * exp(lam) / (exp(lam) - 1) ** 2) < 1e-6
    print("    回火核：和=(1-e^-lam)^-beta，平均滞后=beta/(e^lam-1)，方差=beta*e^lam/(e^lam-1)^2  ✓")
    # beta=1, lambda>0 是几何（指数）核
    raw = gl_raw(1.0, 50, 0.3)
    assert np.allclose(raw[1:] / raw[:-1], exp(-0.3))
    print("    beta=1, lambda>0: 相邻权重比恒为 e^-lambda（指数核）  ✓")


def check_p2_bridge_projection() -> None:
    print("P2  端点桥 = 增量去均值后的累加；范数界")
    rng = np.random.default_rng(0)
    for a, b in ((29, 102), (10, 35), (3, 100)):
        observed = np.ones(FRAMES, dtype=bool)
        observed[a + 1 : b] = False
        residual = rng.normal(size=FRAMES)
        correction = endpoint_bridge(residual, observed)
        segment = residual[a + 1 : b + 1]
        length = b - a
        centered = segment - segment.mean()
        assert np.allclose(correction[a + 1 : b], np.cumsum(centered)[:-1])
        assert abs(np.cumsum(centered)[-1]) < 1e-10          # 右端点处为零
        assert np.allclose(correction[observed], 0.0)         # 已观测帧不受影响
        offsets = np.arange(1, length)
        l2_bound = np.sqrt(offsets * (length - offsets) / length) * np.linalg.norm(segment)
        sup_bound = 2 * offsets * (length - offsets) / length * np.abs(segment).max()
        assert np.all(np.abs(correction[a + 1 : b]) <= l2_bound + 1e-9)
        assert np.all(np.abs(correction[a + 1 : b]) <= sup_bound + 1e-9)
        # 幂等：对去均值后的增量再做一次桥，结果不变
        twice = residual.copy()
        twice[a + 1 : b + 1] = centered
        assert np.allclose(endpoint_bridge(twice, observed), correction)
    print("    三个随机缺口全部通过（恒等式、端点为零、L2 界、L∞ 界、幂等）  ✓")


def check_p3_null_space_and_blind_zones() -> None:
    print("P3  零空间与均匀窗的结构性退化")
    observed = np.ones(FRAMES, dtype=bool)
    observed[30:102] = False
    constant = np.full(FRAMES, 0.37)
    assert np.allclose(endpoint_bridge(constant, observed), 0.0)
    print("    缺口内为常数的增量 -> 修正量恒为零  ✓")

    print("    均匀窗盲区比例（缺口内看不到任何可见增量的帧 / 缺口帧数）")
    print("    缺口   9帧窗   33帧窗   97帧窗")
    for gap in (24, 36, 48, 60, 72, 84, 96):
        start = (FRAMES - gap) // 2
        a, b = start - 1, start + gap
        row = []
        for window in (9, 33, 97):
            blind = 0
            for t in range(start, start + gap):
                left = (t - a) <= window - 1 and a >= 1
                right = (b + 1 - t) <= window - 1 and b + 1 <= FRAMES - 1
                blind += not (left or right)
            closed = max(0, gap + 3 - 2 * window)
            assert blind == closed, (gap, window, blind, closed)
            row.append(f"{blind / gap:6.1%}")
        print(f"    {gap:3d}   " + "   ".join(row))

    print("    97 帧均匀窗：缺口内特征取值完全相同的帧占比（对所有合法起点平均）")
    rng = np.random.default_rng(1)
    uniform = np.full(K, 1.0 / K)
    power_law = gl_raw(0.5, K) / gl_raw(0.5, K).sum()
    for gap in (24, 48, 72, 96):
        dead_fraction = []
        uniform_energy = []
        power_energy = []
        for start in range(1, FRAMES - gap):
            observed = np.ones(FRAMES, dtype=bool)
            observed[start : start + gap] = False
            a, b = start - 1, start + gap
            motion = np.cumsum(rng.normal(size=FRAMES))
            linear = motion.copy()
            linear[start : start + gap] = np.interp(
                np.arange(start, start + gap), [a, b], [motion[a], motion[b]]
            )
            increments = np.zeros(FRAMES)
            increments[1:] = linear[1:] - linear[:-1]
            visible = np.zeros(FRAMES)
            visible[1:] = observed[1:] * observed[:-1]
            inside = slice(start, start + gap)
            feature_uniform = masked_bidirectional(increments, visible, uniform)[inside]
            feature_power = masked_bidirectional(increments, visible, power_law)[inside]
            # 窗口覆盖整段序列的帧：t-96<=1 且 t+96>=127
            covered = [(t - (K - 1) <= 1) and (t + (K - 1) >= FRAMES - 1) for t in range(start, start + gap)]
            dead_fraction.append(np.mean(covered))
            if any(covered):
                values = feature_uniform[np.asarray(covered)]
                assert np.ptp(values) < 1e-9
            uniform_energy.append(np.abs(endpoint_bridge(masked_bidirectional(increments, visible, uniform) - increments, observed)).max())
            power_energy.append(np.abs(endpoint_bridge(masked_bidirectional(increments, visible, power_law) - increments, observed)).max())
        print(
            f"    缺口 {gap:2d}: 常数帧占比={np.mean(dead_fraction):5.1%}"
            f"   本通道桥修正量中位数：均匀97={np.median(uniform_energy):.2e}，GL(0.5)={np.median(power_energy):.2e}"
        )
    print("    （随机游走合成信号，仅说明结构；真实数据上的比例由任务 T0.3 测量）")


def main() -> None:
    check_p1_closed_forms()
    check_p2_bridge_projection()
    check_p3_null_space_and_blind_zones()
    print("全部核对通过。")


if __name__ == "__main__":
    main()
