"""First-order TSK fuzzy inference engine (Sections 3.2-3.3 of the paper).

The engine maps the market vector
``x_t = [Dev_Price, Dev_Volume, CLV]`` to ``Y_attr in [0, 100]`` with

* Gaussian antecedents ``mu_ik = exp(-0.5 ((x_k - c_ik) / max(sigma_ik, delta))^2)``
  (Eq. (5), ``delta = 1e-4`` numerical width floor),
* product firing strengths ``w_k = prod_i mu_ik`` (Eq. (6)),
* weighted-average defuzzification (Eq. (8)) with the neutral ``50.0`` default
  when the total firing mass drops below ``1e-8`` and a ``[0, 100]`` clamp.

Rule centres come from **subtractive clustering** (influence radius 0.5 on
min-max normalised inputs, Chilengui-Mousseau potential update); the number of
rules is fixed at ``K = 5`` after the sensitivity scan. Consequents are solved
by ridge least squares (Eq. (9)); widths are refined by gradient descent using
the closed-form, numerically stable premise gradient implemented here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np

__all__ = [
    "subtractive_clustering",
    "TSKFuzzySystem",
    "DEFAULT_TSK_CONFIG",
]


DEFAULT_TSK_CONFIG: dict = {
    "n_rules": 5,
    "cluster_radius": 0.5,
    "cluster_ratio": 0.5,
    "ridge_lambda": 1e-6,
    "sigma_floor": 1e-4,
    "firing_floor": 1e-8,
    "neutral_score": 50.0,
    "score_scale": 100.0,
    "premise_lr": 1e-3,
    "epochs": 200,
    "grad_clip": 5.0,
    "seed": 0,
}


# ------------------------------------------------------- subtractive clustering
def subtractive_clustering(
    X: np.ndarray,
    radius: float = 0.5,
    ratio: float = 0.5,
    max_rules: int = 5,
) -> np.ndarray:
    """Rule centres via subtractive clustering on min-max normalised inputs.

    Classic Chilengui-Mousseau scheme: potential ``P_i = sum_j exp(-alpha ||x_i - x_j||^2)``
    with ``alpha = 4 / radius^2``; iteratively select the maximum-potential point as a
    centre and squash the remaining potentials by the ``ratio`` factor. The radius is
    shrunk automatically until at least ``max_rules`` centres emerge, then the
    top-``max_rules`` centres by potential are returned (original input scale).
    """
    X = np.asarray(X, dtype=float)
    X = X[np.isfinite(X).all(axis=1)]
    if X.shape[0] == 0:
        raise ValueError("subtractive_clustering requires finite data")
    lo = X.min(axis=0)
    hi = X.max(axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    Z = (X - lo) / span

    r = float(radius)
    centers: list[np.ndarray] = []
    while r > 1e-3:
        alpha = 4.0 / (r * r)
        beta = 4.0 / ((1.25 * r) ** 2)
        dist2 = np.sum((Z[:, None, :] - Z[None, :, :]) ** 2, axis=2)
        potential = np.exp(-alpha * dist2).sum(axis=1)
        candidate_count = 0
        while candidate_count < max_rules:
            k = int(np.argmax(potential))
            if potential[k] <= 0.0:
                break
            centers.append(X[k].copy())
            candidate_count += 1
            potential = _squash(potential, Z, k, alpha_sq=beta, ratio=ratio)
            potential[k] = 0.0
        if candidate_count >= max_rules:
            break
        r *= 0.7
    if len(centers) == 0:
        centers = [X.mean(axis=0)]
    return np.asarray(centers[:max_rules], dtype=float)


def _squash(potential: np.ndarray, Z: np.ndarray, k: int, alpha_sq: float, ratio: float) -> np.ndarray:
    """Potential update after accepting centre ``k`` (Chilengui-Mousseau)."""
    d2 = np.sum((Z - Z[k]) ** 2, axis=1)
    factor = ratio + np.exp(-alpha_sq * d2)
    return np.maximum(potential - potential[k] * factor, 0.0)


# ---------------------------------------------------------------- TSK engine
class TSKFuzzySystem:
    """First-order TSK system with ``K`` Gaussian rules over ``n_inputs`` inputs."""

    def __init__(
        self,
        n_inputs: int,
        n_rules: int = 5,
        sigma_floor: float = 1e-4,
        firing_floor: float = 1e-8,
        neutral_score: float = 50.0,
        score_scale: float = 100.0,
        clip_rule_outputs: bool = True,
        input_scaling: str = "raw",
        feature_names=None,
    ) -> None:
        self.n_inputs = int(n_inputs)
        self.n_rules = int(n_rules)
        self.sigma_floor = float(sigma_floor)
        self.firing_floor = float(firing_floor)
        self.neutral_score = float(neutral_score)
        self.score_scale = float(score_scale)
        # Bounding each rule output to the score domain at inference makes
        # Proposition 1's premise C = max|y_k - Y_raw| <= 100 hold by
        # construction (both terms then lie in [0, score_scale]), for any ridge
        # value and any amount of input extrapolation. Training gradients are
        # unaffected because predict()/premise_gradients() stay unclipped.
        self.clip_rule_outputs = bool(clip_rule_outputs)
        # ``input_scaling`` = "minmax" min-max scales every input to [0, 1] using
        # the *training* bounds before the antecedents and consequents are built,
        # which is required when heterogeneous inputs (e.g. a percentile-style
        # price deviation of ~+-10 next to a unitless fundamental ratio) share one
        # vector. "raw" (default) is the identity map and leaves every existing
        # model unchanged.
        self.input_scaling = str(input_scaling)
        self.scale_lo = np.zeros(self.n_inputs)
        self.scale_span = np.ones(self.n_inputs)
        self.feature_names = list(feature_names) if feature_names else [
            f"x{k}" for k in range(self.n_inputs)
        ]
        self.centers = np.zeros((self.n_rules, self.n_inputs))
        self.sigmas = np.ones((self.n_rules, self.n_inputs))
        self.consequents = np.zeros((self.n_rules, self.n_inputs + 1))
        self.training_metadata: dict = {}

    # ------------------------------------------------------------ antecedents
    def _apply_scaling(self, X: np.ndarray) -> np.ndarray:
        """Map inputs through the train-fitted min-max transform (identity if raw)."""
        array = np.asarray(X, dtype=float)
        return (array - self.scale_lo) / self.scale_span

    def init_from_data(self, X, radius: float = 0.5, ratio: float = 0.5) -> "TSKFuzzySystem":
        """Fit the input scaling, then centres (subtractive clustering) and widths.

        The transform is estimated on this training matrix only; centres and widths
        are computed in the *scaled* space, and ``rule_outputs`` / ``design_matrix``
        also consume scaled inputs, so antecedents and consequents stay consistent.
        """
        array = np.asarray(X, dtype=float)
        if self.input_scaling == "minmax":
            self.scale_lo = np.nanmin(array, axis=0)
            span = np.nanmax(array, axis=0) - self.scale_lo
            self.scale_span = np.where(span > 0, span, 1.0)
        else:
            self.scale_lo = np.zeros(self.n_inputs)
            self.scale_span = np.ones(self.n_inputs)
        work = self._apply_scaling(array)
        self.centers = subtractive_clustering(
            work, radius=radius, ratio=ratio, max_rules=self.n_rules
        )
        spread = np.nanstd(work, axis=0)
        span = np.nanmax(work, axis=0) - np.nanmin(work, axis=0)
        width = np.maximum(0.75 * np.where(spread > 0, spread, span), 4.0 * self.sigma_floor)
        self.sigmas = np.tile(width, (self.n_rules, 1))
        return self

    # --------------------------------------------------------------- inference
    def _scaled(self, X: np.ndarray):
        """``((x - c) / sigma, sigma)`` with the delta floor applied (Eq. (5))."""
        sigma = np.maximum(self.sigmas, self.sigma_floor)
        diff = self._apply_scaling(X)[:, None, :] - self.centers[None, :, :]
        return diff / sigma[None, :, :], sigma

    def _log_firing(self, X: np.ndarray) -> np.ndarray:
        """Log firing strengths ``log w_k`` (log space avoids underflow)."""
        scaled, _ = self._scaled(np.asarray(X, dtype=float))
        return -0.5 * np.sum(scaled ** 2, axis=2)

    def _normalize(self, log_w: np.ndarray) -> np.ndarray:
        shifted = log_w - log_w.max(axis=1, keepdims=True)
        weights = np.exp(shifted)
        return weights / weights.sum(axis=1, keepdims=True)

    def firing_strengths(self, X, normalized: bool = False) -> np.ndarray:
        """Raw (or normalised) firing strengths, shape ``(N, K)``."""
        log_w = self._log_firing(np.asarray(X, dtype=float))
        return np.exp(log_w) if not normalized else self._normalize(log_w)

    def rule_outputs(self, X: np.ndarray) -> np.ndarray:
        """Consequent outputs ``y_k = beta_0k + beta_k . x`` (Eq. (4) THEN part)."""
        array = self._apply_scaling(np.asarray(X, dtype=float))
        return self.consequents[:, 0][None, :] + array @ self.consequents[:, 1:].T

    def _forward(self, X: np.ndarray):
        array = np.asarray(X, dtype=float)
        wn = self._normalize(self._log_firing(array))
        f = self.rule_outputs(array)
        return np.sum(wn * f, axis=1), wn, f

    def predict(self, X) -> np.ndarray:
        """Raw defuzzified output (Eq. (8)) -- no clamp / neutral handling."""
        return self._forward(np.asarray(X, dtype=float))[0]

    def score(self, X) -> np.ndarray:
        """Algorithm-1 scoring: rule-output bound + neutral default + clamp.

        Rule outputs are bounded to the score domain before defuzzification when
        ``clip_rule_outputs`` is set (default), so the antecedent of
        Proposition 1 holds for the deployed model.
        """
        array = np.asarray(X, dtype=float)
        log_w = self._log_firing(array)
        absolute = np.exp(log_w).sum(axis=1)
        if self.clip_rule_outputs:
            weights = self._normalize(log_w)
            bounded = np.clip(self.rule_outputs(array), 0.0, self.score_scale)
            raw = np.sum(weights * bounded, axis=1)
        else:
            raw = self.predict(array)
        raw = np.where(absolute < self.firing_floor, self.neutral_score, raw)
        return np.clip(raw, 0.0, self.score_scale)

    # --------------------------------------------------------------- learning
    def design_matrix(self, X) -> np.ndarray:
        """ANFIS LS design matrix: row ``t`` stacks ``w_bar_k * [1, x_t]`` (Eq. (9))."""
        array = self._apply_scaling(np.asarray(X, dtype=float))
        wn = self._normalize(self._log_firing(array))
        augmented = np.hstack([np.ones((len(array), 1)), array])
        return wn[:, :, None] * augmented[:, None, :]

    def fit_consequents(self, X, y, ridge: float = 1e-6) -> None:
        """Ridge least-squares solve for all rule consequents (Eq. (9))."""
        rows = self.design_matrix(X).reshape(len(X), -1)
        target = np.asarray(y, dtype=float)
        gram = rows.T @ rows + ridge * np.eye(rows.shape[1])
        solution = np.linalg.solve(gram, rows.T @ target)
        self.consequents = solution.reshape(self.n_rules, self.n_inputs + 1)

    def premise_gradients(self, X, y) -> dict[str, np.ndarray]:
        """Closed-form gradients of ``E = 0.5 sum (yhat - y)^2``.

        Uses the numerically stable identity
        ``d yhat / d log w_k = w_bar_k (f_k - yhat)`` (see the paper's Eq. (11)
        discussion), giving

        * ``dE / d log sigma_ik = e_t * w_bar_k (f_k - yhat) * scaled_ik^2``
        * ``dE / d c_ik         = -e_t * w_bar_k (f_k - yhat) * scaled_ik / sigma_ik``
        """
        array = np.asarray(X, dtype=float)
        yhat, wn, f = self._forward(array)
        error = (yhat - np.asarray(y, dtype=float))[:, None]        # (N, 1)
        coeff = error * wn * (f - yhat[:, None])                    # (N, K)
        scaled, sigma = self._scaled(array)
        grad_log_sigma = np.einsum("nk,nki->ki", coeff, scaled ** 2)
        # d log w_k / d c_ik = + scaled_ik / sigma_ik  (since d scaled / d c = -1/sigma)
        grad_centers = np.einsum("nk,nki->ki", coeff, scaled / sigma[None, :, :])
        return {"log_sigma": grad_log_sigma, "centers": grad_centers}

    @property
    def n_free_params(self) -> int:
        """Free trainable parameters: Gaussian widths + consequents.

        ``K * n_inputs`` widths plus ``K * (n_inputs + 1)`` consequent
        coefficients (intercept included) -- the antecedent centres are fixed by
        subtractive clustering. For the paper's configuration this is
        ``5*3 + 5*4 = 35``.
        """
        return self.n_rules * self.n_inputs + self.n_rules * (self.n_inputs + 1)

    @property
    def n_total_params(self) -> int:
        """All parameters including cluster-fixed centres."""
        return 3 * self.n_rules * self.n_inputs + self.n_rules

    # ------------------------------------------------------------ persistence
    def to_dict(self) -> dict:
        return {
            "model": "TSKFuzzySystem",
            "version": 3,
            "n_inputs": self.n_inputs,
            "n_rules": self.n_rules,
            "sigma_floor": self.sigma_floor,
            "score_scale": self.score_scale,
            "neutral_score": self.neutral_score,
            "clip_rule_outputs": self.clip_rule_outputs,
            "input_scaling": self.input_scaling,
            "scale_lo": self.scale_lo.tolist(),
            "scale_span": self.scale_span.tolist(),
            "feature_names": self.feature_names,
            "centers": self.centers.tolist(),
            "sigmas": self.sigmas.tolist(),
            "consequents": self.consequents.tolist(),
            "training_metadata": self.training_metadata,
        }

    @classmethod
    def from_dict(cls, payload: Mapping) -> "TSKFuzzySystem":
        model = cls(
            n_inputs=int(payload["n_inputs"]),
            n_rules=int(payload["n_rules"]),
            sigma_floor=float(payload.get("sigma_floor", 1e-4)),
            neutral_score=float(payload.get("neutral_score", 50.0)),
            score_scale=float(payload.get("score_scale", 100.0)),
            clip_rule_outputs=bool(payload.get("clip_rule_outputs", True)),
            input_scaling=str(payload.get("input_scaling", "raw")),
            feature_names=payload.get("feature_names"),
        )
        model.centers = np.asarray(payload["centers"], dtype=float)
        model.sigmas = np.asarray(payload["sigmas"], dtype=float)
        model.consequents = np.asarray(payload["consequents"], dtype=float)
        if "scale_lo" in payload:
            model.scale_lo = np.asarray(payload["scale_lo"], dtype=float)
            model.scale_span = np.asarray(payload["scale_span"], dtype=float)
        model.training_metadata = dict(payload.get("training_metadata", {}))
        return model

    def save(self, path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path) -> "TSKFuzzySystem":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
