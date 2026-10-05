"""Anti-frontrunning stochastic masking (Section 3.4 of the paper).

Within the Pre-AGM window ``[t_AGM - M_pre, t_AGM - 1]`` the raw firing strengths
are perturbed as ``w_masked = w * (1 + xi)`` with ``xi ~ N(0, (alpha/2)^2)``
truncated to ``[-alpha, +alpha]`` (``alpha = 0.05``), before defuzzification
(Eqs. (14)-(15)). Proposition 1 bounds the induced bias by ``C sigma_xi^2``
(second-order Taylor); the helper :func:`expected_masked_output` implements the
exact first-order expansion used in the proof sketch (Eq. (16)).
"""

from __future__ import annotations

import numpy as np

from src.TSK_engine import TSKFuzzySystem

__all__ = [
    "truncated_gaussian",
    "apply_mask",
    "masked_firing_scores",
    "masked_scores",
    "expected_masked_output",
    "prop1_bias_bound",
]


def truncated_gaussian(size, alpha: float, rng: np.random.Generator) -> np.ndarray:
    """Zero-mean Gaussian with std ``alpha/2``, truncated to ``[-alpha, alpha]``."""
    sigma = alpha / 2.0
    draw = rng.normal(0.0, sigma, size=size)
    while True:
        outside = np.abs(draw) > alpha
        if not outside.any():
            return draw
        draw[outside] = rng.normal(0.0, sigma, size=int(outside.sum()))


def apply_mask(firing: np.ndarray, alpha: float, rng: np.random.Generator) -> np.ndarray:
    """Perturb raw firing strengths ``w -> w (1 + xi)`` (Eq. (14), never negative)."""
    xi = truncated_gaussian(firing.shape, alpha, rng)
    return np.maximum(firing * (1.0 + xi), 0.0)


def masked_firing_scores(
    model: TSKFuzzySystem,
    X: np.ndarray,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Algorithm-1 scores with masked rule weights: neutral default + clamp.

    Rule outputs are bounded to the score domain whenever the model does so on
    its unmasked path, keeping the deployed masked and unmasked scorers
    consistent and Proposition 1's premise valid for both.
    """
    rng = rng or np.random.default_rng()
    array = np.asarray(X, dtype=float)
    scaled, _ = model._scaled(array)
    firing = np.exp(-0.5 * np.sum(scaled ** 2, axis=2))
    masked = apply_mask(firing, alpha, rng)
    f = model.rule_outputs(array)
    if model.clip_rule_outputs:
        f = np.clip(f, 0.0, model.score_scale)
    total = masked.sum(axis=1)
    num = np.sum(masked * f, axis=1)
    raw = np.where(
        total > model.firing_floor,
        num / np.where(total > 0, total, 1.0),
        model.neutral_score,
    )
    return np.clip(raw, 0.0, model.score_scale)


def masked_scores(model: TSKFuzzySystem, X, pre_agm=None, alpha: float = 0.05, rng=None) -> np.ndarray:
    """Deployed scoring path: mask rule weights only where ``I_PreAGM = 1``.

    ``alpha = 0`` -- or a flag vector that is everywhere zero -- reproduces
    :meth:`TSKFuzzySystem.score` exactly, so one call site can report both the
    research path and the deployed path of Section 3.4.
    """
    array = np.asarray(X, dtype=float)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    base = model.score(array)
    if alpha <= 0.0:
        return base
    flags = (
        np.ones(len(array), dtype=bool)
        if pre_agm is None
        else np.asarray(pre_agm, dtype=float).reshape(-1) > 0.0
    )
    if not flags.any():
        return base
    out = base.copy()
    out[flags] = masked_firing_scores(model, array[flags], alpha=alpha, rng=rng)
    return out


def expected_masked_output(firing: np.ndarray, rule_outputs: np.ndarray) -> np.ndarray:
    """First-order expansion of E[Y_masked] (Eq. (16)).

    ``E[Y_masked] approx Y_raw - sigma_xi^2 * sum_k w_bar_k^2 (y_k - Y_raw)``
    with ``sigma_xi = alpha / 2``. Returns ``E[Y_masked]`` per observation.
    """
    sigma_xi2 = 0.025 ** 2  # alpha = 0.05 by default in the protocol
    total = firing.sum(axis=1, keepdims=True)
    w_bar = firing / np.where(total > 0, total, 1.0)
    y_raw = np.sum(w_bar * rule_outputs, axis=1, keepdims=True)
    correction = sigma_xi2 * np.sum(w_bar ** 2 * (rule_outputs - y_raw), axis=1, keepdims=True)
    return (y_raw - correction).ravel()


def prop1_bias_bound(alpha: float = 0.05, c: float = 100.0) -> float:
    """Proposition 1 worst-case bias ``C * (alpha/2)^2`` on the [0, 100] scale."""
    return c * (alpha / 2.0) ** 2
