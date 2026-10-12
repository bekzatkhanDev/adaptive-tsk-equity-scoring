"""Truncation-calibrated adaptive alpha (Referee M2 / Report Analysis 3).

``truncated_gaussian`` draws ``xi ~ N(0, (alpha/2)^2)`` truncated to
``+/-alpha = +/-2 sigma``, whose realised variance is ``kappa * (alpha/2)^2``
with ``kappa = 0.7738`` (exactly ``1 - 4*phi(2)/(2*Phi(2)-1)``). The previously
published inversion ``alpha = 2*eps/sqrt(S)`` ignored that factor, so realised
noise SD landed at ``~0.88 * eps`` even where the cap does not bind (the
reported 1.76 against the 2.0 target). The deployed scheme now inverts the
*truncation-calibrated* identity (``src/masking.py:adaptive_alpha``:
``alpha = 2*eps/(sqrt(kappa)*sqrt(S))``); this script verifies, on the real
Pre-AGM panels, both inversions:

* realised output-noise SD vs the target ``epsilon`` (uncapped vs capped rows);
* signed/absolute deviation (bias terminology per referee M2);
* a Monte-Carlo check of the first-order expansion at large alpha
  (alpha in 0.05..1.0): empirical vs Eq. (17) predicted variance, and
  empirical mean bias vs the Eq. (16) second-order prediction -- the
  validity region of Proposition 1's expansion.

Example
-------
python experiments/truncation_correction.py --draws 400
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.data_loader import FEATURE_COLUMNS, load_config
from src.masking import (
    TRUNCATION_VARIANCE_FACTOR,
    adaptive_alpha,
    expected_masked_output,
    masking_variance,
    truncated_gaussian,
)
from src.TSK_engine import TSKFuzzySystem
from experiments.adaptive_masking import _panel, _firing

TRUNC_FACTOR = TRUNCATION_VARIANCE_FACTOR


def published_alpha(firing, rule_outputs, epsilon=2.0, alpha_max=1.0):
    """The previously published (uncalibrated) inversion: ignores kappa."""
    total = firing.sum(axis=1, keepdims=True)
    w_bar = firing / np.maximum(total, 1e-12)
    y_raw = np.sum(w_bar * rule_outputs, axis=1, keepdims=True)
    s = np.sum(w_bar ** 2 * (rule_outputs - y_raw) ** 2, axis=1)
    req = 2.0 * epsilon / np.sqrt(np.maximum(s, 1e-12))
    alphas = np.minimum(req, alpha_max)
    return np.where(s <= 1e-8, 0.0, alphas).ravel()


def rowwise_noise(model, X, alphas, draws, seed):
    """Realised masked outputs over ``draws`` independent masks (per-row alpha)."""
    firing, f = _firing(model, X)
    rng = np.random.default_rng(seed)
    outs = np.empty((draws, len(X)))
    for d in range(draws):
        for i in range(len(X)):
            xi = truncated_gaussian((model.n_rules,), float(alphas[i]), rng)
            w = np.maximum(firing[i] * (1.0 + xi), 0.0)
            tot = w.sum()
            outs[d, i] = (float(np.sum(w * f[i]) / tot)
                          if tot > model.firing_floor else model.neutral_score)
    return outs, firing, f


def predicted_sd(firing, f, alphas):
    """Eq. (17) per-row predicted *realised* output SD for the given alphas."""
    total = firing.sum(axis=1, keepdims=True)
    w_bar = firing / np.maximum(total, 1e-12)
    y_raw = np.sum(w_bar * f, axis=1, keepdims=True)
    s = np.sum(w_bar ** 2 * (f - y_raw) ** 2, axis=1)
    return np.sqrt(TRUNC_FACTOR) * 0.5 * alphas * np.sqrt(np.maximum(s, 1e-12))



def main() -> None:
    ap = argparse.ArgumentParser(description="truncation-corrected alpha (analysis 3)")
    ap.add_argument("--draws", type=int, default=400)
    ap.add_argument("--epsilon", type=float, default=2.0)
    ap.add_argument("--alpha-max", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    cfg = load_config()
    model = TSKFuzzySystem.load(Path(cfg["paths"]["models"]) / "panel_anfis.json")
    train, test = _panel(cfg)
    x = np.vstack([
        train[list(FEATURE_COLUMNS)].to_numpy(dtype=float),
        test[list(FEATURE_COLUMNS)].to_numpy(dtype=float),
    ])
    firing, f = _firing(model, x)
    y_clean = model.score(x)

    report: dict = {"truncation_variance_factor": TRUNC_FACTOR,
                    "epsilon": args.epsilon, "draws": args.draws}
    for name, alphas in [
        ("published", published_alpha(firing, f, args.epsilon, args.alpha_max)),
        ("calibrated", adaptive_alpha(firing, f, args.epsilon, args.alpha_max)),
    ]:
        cap_rate = float(np.mean(alphas >= args.alpha_max - 1e-12))
        outs, _, _ = rowwise_noise(model, x, alphas, args.draws, args.seed)
        noise = outs - y_clean[None, :]
        sd_realised = noise.std(axis=0)
        pred = predicted_sd(firing, f, alphas)
        uncapped = alphas < args.alpha_max - 1e-12
        report[name] = {
            "alpha_median": float(np.median(alphas)),
            "alpha_p90": float(np.quantile(alphas, 0.9)),
            "cap_rate": cap_rate,
            "realised_sd_median": float(np.median(sd_realised)),
            "realised_sd_median_uncapped": float(np.median(sd_realised[uncapped])),
            "target": args.epsilon,
            "ratio_to_target": float(np.median(sd_realised[uncapped]) / args.epsilon),
            "pred_vs_realised_median": float(
                np.median(sd_realised[uncapped] / np.maximum(pred[uncapped], 1e-12))),
            "signed_bias_mean": float(noise.mean()),
            "mean_abs_deviation": float(np.abs(noise).mean()),
            "noise_sd_pooled": float(noise.std()),
        }

    # Monte-Carlo remainder check at large alpha (referee A.3, third bullet):
    # empirical vs first-order predicted variance on the actual firing rows.
    remainder = []
    for alpha in (0.05, 0.3, 0.6, 1.0):
        outs = np.empty((args.draws, len(x)))
        rng = np.random.default_rng(args.seed)
        for d in range(args.draws):
            xi = truncated_gaussian(firing.shape, alpha, rng)
            w = np.maximum(firing * (1.0 + xi), 0.0)
            tot = w.sum(axis=1)
            outs[d] = np.where(tot > model.firing_floor,
                               np.sum(w * f, axis=1) / np.maximum(tot, 1e-12),
                               model.neutral_score)
        emp = outs.var(axis=0)
        pred_v = masking_variance(firing, f, alpha=alpha)
        m = pred_v > 1e-10
        remainder.append({
            "alpha": alpha,
            "sigma_xi": alpha / 2.0,
            "median_emp_over_pred": float(np.median(emp[m] / pred_v[m])),
            "mean_emp_over_pred": float(np.mean(emp[m] / pred_v[m])),
        })
    report["remainder_check"] = remainder

    # Monte-Carlo validity of the Eq. (16) second-order bias expansion at
    # large alpha (Prop. 1 remainder, Referee M2): empirical mean bias vs
    # the expansion's prediction on the real firing rows.
    total = firing.sum(axis=1, keepdims=True)
    w_bar = firing / np.maximum(total, 1e-12)
    y_ref = np.sum(w_bar * f, axis=1)
    bias_validity = []
    for alpha in (0.05, 0.3, 0.6, 1.0):
        rng = np.random.default_rng(args.seed)
        outs = np.empty((args.draws, len(x)))
        for d in range(args.draws):
            xi = truncated_gaussian(firing.shape, alpha, rng)
            w = np.maximum(firing * (1.0 + xi), 0.0)
            tot = w.sum(axis=1)
            outs[d] = np.where(tot > model.firing_floor,
                               np.sum(w * f, axis=1) / np.maximum(tot, 1e-12),
                               model.neutral_score)
        emp_bias = outs.mean(axis=0) - y_ref
        pred_bias = expected_masked_output(firing, f, alpha=alpha) - y_ref
        m = np.abs(pred_bias) > 1e-6
        bias_validity.append({
            "alpha": alpha,
            "sigma_xi_realised": float(np.sqrt(TRUNC_FACTOR) * alpha / 2.0),
            "median_emp_over_pred": float(np.median(emp_bias[m] / pred_bias[m])),
            "mean_abs_emp": float(np.abs(emp_bias).mean()),
            "mean_abs_pred": float(np.abs(pred_bias).mean()),
            "max_abs_residual": float(np.abs(emp_bias[m] - pred_bias[m]).max()),
        })
    report["bias_validity_check"] = bias_validity

    logs = Path(cfg["paths"]["logs"])
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "truncation_correction.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    for name in ("published", "calibrated"):
        r = report[name]
        print(f"{name:10s} alpha_med={r['alpha_median']:.3f} cap={r['cap_rate']:.0%} "
              f"realised_sd={r['realised_sd_median']:.3f} "
              f"(uncapped {r['realised_sd_median_uncapped']:.3f} = "
              f"{r['ratio_to_target']:.3f}x target) bias={r['signed_bias_mean']:+.3f} "
              f"MAD={r['mean_abs_deviation']:.3f}")
    for row in report["remainder_check"]:
        print(f"remainder alpha={row['alpha']:.2f} (sigma_xi={row['sigma_xi']:.2f}): "
              f"emp/pred median={row['median_emp_over_pred']:.3f} "
              f"mean={row['mean_emp_over_pred']:.3f}")
    print(f"\nfull report: {logs / 'truncation_correction.json'}")


if __name__ == "__main__":
    main()

