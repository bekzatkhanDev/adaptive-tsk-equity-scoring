"""Anti-frontrunning stochastic masking (Section 3.4 of the paper).

Within the Pre-AGM window ``[t_AGM - M_pre, t_AGM - 1]`` the raw firing strengths
are perturbed as ``w_masked = w * (1 + xi)`` with ``xi ~ N(0, (alpha/2)^2)``
truncated to ``[-alpha, +alpha]`` (``alpha = 0.05``), before defuzzification
(Eqs. (14)-(15)).

The +/-alpha truncation sits at exactly +/-2 sigma for every alpha, so the
draw's *realised* variance is ``kappa * (alpha/2)^2`` with
``kappa = TRUNCATION_VARIANCE_FACTOR = 0.7738``. Every variance identity
(Eq. (17)) and the adaptive inversion (Eq. (18)) below use that realised
variance, and the adaptive scheme is calibrated to it so uncapped rows attain
the target noise ``epsilon`` (Referee M2 fix). Proposition 1's second-order
bias expansion (Eq. (16)) is implemented by :func:`expected_masked_output`;
its remainder is validated by Monte Carlo in
``experiments/truncation_correction.py``.
"""

from __future__ import annotations

import hashlib

import numpy as np

from src.TSK_engine import TSKFuzzySystem

__all__ = [
    "truncated_gaussian",
    "apply_mask",
    "masked_firing_scores",
    "masked_scores",
    "expected_masked_output",
    "prop1_bias_bound",
    "unlimited_query_bound",
    "effective_rules",
    "masking_variance",
    "masking_amplification",
    "adaptive_alpha",
    "adaptive_masked_firing_scores",
    "keyed_masked_scores",
    "TRUNCATION_VARIANCE_FACTOR",
]

# xi ~ N(0, (alpha/2)^2) truncated to [-alpha, +alpha] = +/-2 sigma: the exact
# retained-variance factor is 1 - 2a*phi(a)/(2*Phi(a)-1) with a = 2. The
# truncation sits at a constant 2 sigma for every alpha, so kappa is scale-free;
# all variance identities and the adaptive inversion use the realised variance
# kappa*(alpha/2)^2 (Referee M2: calibrate to the truncated variance).
TRUNCATION_VARIANCE_FACTOR = 1.0 - 4.0 * 0.05399096651318778 / (2.0 * 0.9772498680518208 - 1.0)


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


def keyed_masked_scores(
    model: TSKFuzzySystem,
    X: np.ndarray,
    keys: list[str] | np.ndarray,
    secret: str,
    pre_agm: np.ndarray | None = None,
    alpha: float = 0.05,
) -> np.ndarray:
    """Mask deterministically per secret, asset/date key.

    Repeated queries for the same key receive the same score, preventing
    query averaging. The secret must remain server-side; changing it rotates
    the deterministic mask and defines a new masking epoch.
    """
    array = np.asarray(X, dtype=float)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    row_keys = np.asarray(keys, dtype=str).reshape(-1)
    if len(row_keys) != len(array):
        raise ValueError("keys must have one entry per input row")
    base = model.score(array)
    if alpha <= 0.0:
        return base
    flags = (
        np.ones(len(array), dtype=bool)
        if pre_agm is None
        else np.asarray(pre_agm, dtype=float).reshape(-1) > 0.0
    )
    out = base.copy()
    for index in np.flatnonzero(flags):
        digest = hashlib.blake2b(
            f"{secret}\0{row_keys[index]}".encode("utf-8"), digest_size=16
        ).digest()
        seed = int.from_bytes(digest, byteorder="little", signed=False)
        out[index] = masked_firing_scores(
            model,
            array[index : index + 1],
            alpha=alpha,
            rng=np.random.default_rng(seed),
        )[0]
    return out


def expected_masked_output(
    firing: np.ndarray, rule_outputs: np.ndarray, alpha: float = 0.05
) -> np.ndarray:
    """Second-order expansion of E[Y_masked] (Eq. (16)).

    ``E[Y_masked] ~= Y_raw - sigma_xi^2 * sum_k w_bar_k^2 (y_k - Y_raw)``
    with the realised truncated noise variance
    ``sigma_xi^2 = TRUNCATION_VARIANCE_FACTOR * (alpha/2)^2``. The remainder
    (O(sigma_xi^3) in the paper's Prop. 1 restatement) is measured against
    this prediction by ``experiments/truncation_correction.py``.
    """
    sigma_xi2 = TRUNCATION_VARIANCE_FACTOR * (alpha / 2.0) ** 2
    total = firing.sum(axis=1, keepdims=True)
    w_bar = firing / np.where(total > 0, total, 1.0)
    y_raw = np.sum(w_bar * rule_outputs, axis=1, keepdims=True)
    correction = sigma_xi2 * np.sum(w_bar ** 2 * (rule_outputs - y_raw), axis=1, keepdims=True)
    return (y_raw - correction).ravel()


def unlimited_query_bound(alpha: float = 0.05, n_rules: int = 5) -> dict[str, float]:
    """Formal query-averaging bound for the masking noise (review item 4).

    Each masked response is ``Y_raw + noise`` with ``Var(noise) <= C^2 sigma_xi^2``
    to first order (Proposition 1 linearisation, realised
    ``sigma_xi = sqrt(kappa) * alpha / 2``). An adversary averaging ``Q``
    independent queries for the same ``x_t`` sees standard error
    ``<= C sigma_xi / sqrt(Q)``; the required queries to bring that error
    below ``eps`` score points are ``(C sigma_xi / eps)^2``. With ``C = 100``,
    ``alpha = 0.05``: one query leaves up to 2.2 score points of noise;
    averaging ~20 queries brings it below 0.5 points. Masking is
    therefore *obfuscation with a quantified query budget*, not security
    against an unlimited-query adversary -- the paper's Section 3.4 remark.
    """
    sigma_xi = np.sqrt(TRUNCATION_VARIANCE_FACTOR) * alpha / 2.0
    return {
        "alpha": float(alpha),
        "sigma_xi": float(sigma_xi),
        "single_query_noise_cap": 100.0 * sigma_xi,
        "queries_to_half_point": float((100.0 * sigma_xi / 0.5) ** 2),
        "queries_to_tenth_point": float((100.0 * sigma_xi / 0.1) ** 2),
        "n_rules": int(n_rules),
    }
def prop1_bias_bound(alpha: float = 0.05, c: float = 100.0) -> float:
    """Proposition 1 worst-case bias ``C * (alpha/2)^2`` on the [0, 100] scale.

    A second-order, worst-case envelope (``C = 100`` is the global consequent
    bound, ``sigma_xi = alpha/2`` the untruncated nominal scale); truncation
    only *shrinks* the realised variance, so the envelope stays valid while
    being deliberately loose -- the informative number is the Monte-Carlo
    bias validity measured against Eq. (16)
    (``experiments/truncation_correction.py``).
    """
    return c * (alpha / 2.0) ** 2


def effective_rules(firing: np.ndarray) -> np.ndarray:
    """Effective number of active rules (Eq. (17) context, inverse Herfindahl).

    ``K_eff,t = 1 / sum_k w_bar_k,t^2`` with ``w_bar`` the normalised firing
    strengths. ``K_eff = 1`` means one rule dominates; ``K_eff = K`` means all
    rules contribute equally.
    """
    firing = np.asarray(firing, dtype=float)
    total = firing.sum(axis=1, keepdims=True)
    w_bar = firing / np.where(total > 0, total, 1.0)
    return 1.0 / np.maximum(np.sum(w_bar ** 2, axis=1), 1e-12)


def masking_variance(
    firing: np.ndarray,
    rule_outputs: np.ndarray,
    alpha: float = 0.05,
) -> np.ndarray:
    """First-order masked-output variance of Eq. (17), truncation-calibrated.

    ``Var[Y_masked,t] = sigma_xi^2 * sum_k w_bar_k,t^2 (y_k,t - Y_raw,t)^2``
    with the *realised* truncated noise variance
    ``sigma_xi^2 = TRUNCATION_VARIANCE_FACTOR * (alpha/2)^2``. Validated
    against Monte Carlo on the paper's Pre-AGM panel (median
    empirical/predicted ratio 1.00 at every alpha, cross-row correlation
    0.999; using the untruncated ``(alpha/2)^2`` instead gives 0.77 -- the
    truncation shortfall of Referee M2).
    """
    firing = np.asarray(firing, dtype=float)
    rule_outputs = np.asarray(rule_outputs, dtype=float)
    sigma2 = TRUNCATION_VARIANCE_FACTOR * (alpha / 2.0) ** 2
    total = firing.sum(axis=1, keepdims=True)
    w_bar = firing / np.where(total > 0, total, 1.0)
    y_raw = np.sum(w_bar * rule_outputs, axis=1, keepdims=True)
    return (sigma2 * np.sum(w_bar ** 2 * (rule_outputs - y_raw) ** 2, axis=1)).ravel()


def masking_amplification(firing: np.ndarray, rule_outputs: np.ndarray) -> np.ndarray:
    """Masking amplification factor ``Lambda_t`` of Eq. (17).

    ``Lambda_t = sum_k w_bar_k^2 (y_k - Y_raw)^2 / V_t`` with ``V_t`` the
    weighted consequent dispersion. For uniform weights ``Lambda = 1/K``;
    for concentrated weights (``w_bar_1 -> 1``) ``Lambda -> 0`` and masking
    vanishes: at most one rule's deviation survives, and it is multiplied by
    the vanishing weight of every other rule.
    """
    firing = np.asarray(firing, dtype=float)
    rule_outputs = np.asarray(rule_outputs, dtype=float)
    total = firing.sum(axis=1, keepdims=True)
    w_bar = firing / np.where(total > 0, total, 1.0)
    y_raw = np.sum(w_bar * rule_outputs, axis=1, keepdims=True)
    v = np.sum(w_bar * (rule_outputs - y_raw) ** 2, axis=1)
    num = np.sum(w_bar ** 2 * (rule_outputs - y_raw) ** 2, axis=1)
    return (num / np.maximum(v, 1e-12)).ravel()


def adaptive_alpha(
    firing: np.ndarray,
    rule_outputs: np.ndarray,
    epsilon: float = 2.0,
    alpha_max: float = 1.0,
) -> np.ndarray:
    """Dispersion-adaptive noise scale (Section 3.4.2), truncation-calibrated.

    Inverts the first-order variance identity (Eq. (17)) with the *realised*
    truncated noise variance so the masked output carries variance
    ``epsilon^2`` wherever the cap does not bind (Referee M2)::

        alpha_t = min(alpha_max, 2 * epsilon / (sqrt(kappa) * sqrt(S_t)))

    with ``kappa = TRUNCATION_VARIANCE_FACTOR = 0.7738`` and
    ``S_t = sum_k w_bar_k^2 (y_k - Y_raw)^2``. The previously published
    inversion (kappa omitted) landed the realised SD at
    ``sqrt(kappa) * epsilon ~= 0.88 * epsilon`` even on uncapped rows;
    calibrating removes that shortfall. Since ``alpha_t`` grows as ``S_t``
    shrinks, the cap binds on the *smallest*-``S_t`` rows, where even
    ``alpha_max`` cannot reach the target; the caller reports the binding
    rate.

    When ``S_t <= delta_s`` (numerically zero — all rules agree on the
    consequent), the score is invariant to rule-weight masking and
    ``alpha_t`` is set to 0 rather than dividing by zero.
    """
    firing = np.asarray(firing, dtype=float)
    rule_outputs = np.asarray(rule_outputs, dtype=float)
    total = firing.sum(axis=1, keepdims=True)
    w_bar = firing / np.where(total > 0, total, 1.0)
    y_raw = np.sum(w_bar * rule_outputs, axis=1, keepdims=True)
    s = np.sum(w_bar ** 2 * (rule_outputs - y_raw) ** 2, axis=1)
    req = (2.0 * epsilon
           / (np.sqrt(TRUNCATION_VARIANCE_FACTOR) * np.sqrt(np.maximum(s, 1e-12))))
    alphas = np.minimum(req, alpha_max)
    return np.where(s <= 1e-8, 0.0, alphas).ravel()


def adaptive_masked_firing_scores(
    model: TSKFuzzySystem,
    X: np.ndarray,
    epsilon: float = 2.0,
    alpha_max: float = 1.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Score with per-row dispersion-adaptive masking (Section 3.4.2).

    Returns ``(scores, alphas, predicted_sd)``: the masked scores, the per-row
    ``alpha_t`` from :func:`adaptive_alpha`, and the truncation-calibrated
    Eq. (17) predicted masked-output standard deviation (``epsilon`` wherever
    the cap does not bind, below ``epsilon`` on cap-bound rows).
    """
    rng = rng or np.random.default_rng()
    array = np.asarray(X, dtype=float)
    scaled, _ = model._scaled(array)
    firing = np.exp(-0.5 * np.sum(scaled ** 2, axis=2))
    f = model.rule_outputs(array)
    if model.clip_rule_outputs:
        f = np.clip(f, 0.0, model.score_scale)
    total = firing.sum(axis=1, keepdims=True)
    w_bar = firing / np.where(total > 0, total, 1.0)
    y_raw = np.sum(w_bar * f, axis=1, keepdims=True)
    s = np.sum(w_bar ** 2 * (f - y_raw) ** 2, axis=1)
    alphas = adaptive_alpha(firing, f, epsilon=epsilon, alpha_max=alpha_max)
    pred_sd = np.minimum(
        np.full(len(array), epsilon),
        np.sqrt(TRUNCATION_VARIANCE_FACTOR) * (alphas / 2.0)
        * np.sqrt(np.maximum(s, 1e-12)))
    out = np.empty(len(array))
    for i in range(len(array)):
        xi = truncated_gaussian((model.n_rules,), float(alphas[i]), rng)
        w = np.maximum(firing[i] * (1.0 + xi), 0.0)
        tot = w.sum()
        out[i] = (
            float(np.sum(w * f[i]) / tot)
            if tot > model.firing_floor
            else model.neutral_score
        )
    return np.clip(out, 0.0, model.score_scale), alphas, pred_sd
