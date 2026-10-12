"""Dispersion-adaptive masking validation (Proposition 2 + Section 3.4.2).

Four diagnostics on the Pre-AGM panel: (1) keff rule-concentration
distribution; (2) variance identity check (Monte-Carlo vs truncation-calibrated
Eq. 17 over an alpha grid); (3) adversary SVR reconstruction under fixed vs
adaptive masking; (4) descriptive per-row reconstruction-error comparisons on concentrated
rows, fixed-matched vs adaptive; these are not inferential evidence because
the terciles use one small test-panel draw.

Example: python experiments/adaptive_masking.py --epsilon 2.0 --seed 7
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.svm import SVR

from src.data_loader import FEATURE_COLUMNS, build_panel, load_config
from src.masking import (
    adaptive_alpha,
    adaptive_masked_firing_scores,
    effective_rules,
    masked_firing_scores,
    masking_amplification,
    masking_variance,
    truncated_gaussian,
    keyed_masked_scores,
)
from src.TSK_engine import TSKFuzzySystem


def _panel(cfg):
    train = build_panel(cfg, split="train")
    test = build_panel(cfg, split="test")
    return (train[train["pre_agm"] > 0].reset_index(drop=True),
            test[test["pre_agm"] > 0].reset_index(drop=True))


def _firing(model, X):
    scaled, _ = model._scaled(np.asarray(X, dtype=float))
    firing = np.exp(-0.5 * np.sum(scaled ** 2, axis=2))
    f = model.rule_outputs(np.asarray(X, dtype=float))
    if model.clip_rule_outputs:
        f = np.clip(f, 0.0, model.score_scale)
    return firing, f

def keff_diagnostic(model, X) -> dict:
    firing, f = _firing(model, X)
    keff = effective_rules(firing)
    lam = masking_amplification(firing, f)
    return {
        "n_rows": int(len(X)),
        "keff_median": float(np.median(keff)),
        "keff_mean": float(np.mean(keff)),
        "keff_max": float(np.max(keff)),
        "lambda_median": float(np.median(lam)),
        "note": "K_eff=1 concentrated .. K_eff=K dispersed; Lambda->0 kills masking",
    }


def variance_check(model, X, alphas=(0.01, 0.05, 0.10, 0.20),
                   draws: int = 500, seed: int = 7) -> dict:
    firing, f = _firing(model, X)
    rows: list[dict] = []
    rng = np.random.default_rng(seed)
    for alpha in alphas:
        outs = np.empty((draws, len(X)))
        for d in range(draws):
            xi = truncated_gaussian(firing.shape, alpha, rng)
            w = np.maximum(firing * (1.0 + xi), 0.0)
            tot = w.sum(axis=1)
            outs[d] = np.where(tot > model.firing_floor,
                               np.sum(w * f, axis=1) / np.maximum(tot, 1e-12),
                               model.neutral_score)
        emp = outs.var(axis=0)
        pred = masking_variance(firing, f, alpha=alpha)
        ratio = emp / np.maximum(pred, 1e-12)
        rows.append({
            "alpha": alpha,
            "emp_sd_median": float(np.median(np.sqrt(emp))),
            "pred_sd_median": float(np.median(np.sqrt(pred))),
            "ratio_median": float(np.median(ratio)),
            "corr": float(np.corrcoef(emp, pred)[0, 1]),
        })
    return {"rows": rows, "draws": draws,
            "note": "ratio~=1.00 validates Eq. (17) with the truncation-"
                    "calibrated variance; without kappa the ratio is 0.77"}


def fixed_at_matched_noise(model, x_train, x_test, target_sd: float,
                            alphas=(0.3, 0.6, 1.0), seed: int = 7) -> dict:
    """Fixed-alpha sweep to match the adaptive scheme's realised noise SD.

    Reports per-alpha realised output-noise SD, bias, and SVR R^2 so the
    adaptive scheme can be compared against fixed masking at the same
    effective noise budget (reviewer issue 4.1).
    """
    y_tr = model.score(x_train)
    y_te = model.score(x_test)
    rows: list[dict] = []
    for alpha in alphas:
        rng = np.random.default_rng(seed)
        y_tr_f = masked_firing_scores(model, x_train, alpha=alpha, rng=rng)
        y_te_f = masked_firing_scores(model, x_test, alpha=alpha, rng=rng)
        noise = np.concatenate([y_tr_f - y_tr, y_te_f - y_te])
        svr = SVR(kernel="rbf", C=1.0, gamma="scale")
        svr.fit(x_train, y_tr_f)
        rows.append({
            "alpha": alpha,
            "noise_sd": float(noise.std()),
            "bias_abs": float(np.abs(noise).mean()),
            "r2": float(r2_score(y_te_f, svr.predict(x_test))),
        })
    return {"rows": rows, "target_sd": target_sd,
            "note": "compare fixed rows at ~= target_sd vs adaptive R^2"}


def adversary(model, x_train, x_test, epsilon: float, alpha_max: float,
              seed: int) -> dict:
    y_tr = model.score(x_train)
    y_te = model.score(x_test)
    rng = np.random.default_rng(seed)
    y_tr_fix = masked_firing_scores(model, x_train, alpha=0.05, rng=rng)
    y_te_fix = masked_firing_scores(model, x_test, alpha=0.05, rng=rng)
    y_tr_ad, a_tr, _ = adaptive_masked_firing_scores(
        model, x_train, epsilon=epsilon, alpha_max=alpha_max, rng=rng)
    y_te_ad, a_te, _ = adaptive_masked_firing_scores(
        model, x_test, epsilon=epsilon, alpha_max=alpha_max, rng=rng)
    out = {}
    for name, tr, te in [("unmasked", y_tr, y_te),
                         ("fixed", y_tr_fix, y_te_fix),
                         ("adaptive", y_tr_ad, y_te_ad)]:
        svr = SVR(kernel="rbf", C=1.0, gamma="scale")
        svr.fit(x_train, tr)
        out["r2_" + name] = float(r2_score(te, svr.predict(x_test)))
    out.update({
        "adaptive_alpha_median": float(np.median(np.concatenate([a_tr, a_te]))),
        "adaptive_alpha_cap_rate": float(
            np.mean(np.concatenate([a_tr, a_te]) >= alpha_max - 1e-12)),
        "adaptive_bias_signed": float(
            np.mean(np.concatenate([y_tr_ad, y_te_ad])
                    - np.concatenate([y_tr, y_te]))),
        "adaptive_bias_abs": float(
            np.mean(np.abs(np.concatenate([y_tr_ad, y_te_ad])
                            - np.concatenate([y_tr, y_te])))),
        "epsilon": epsilon, "alpha_max": alpha_max,
    })
    return out


def keyed_adversary(model, x_train, x_test, train_keys, test_keys,
                    secret: str = "audit-secret") -> dict:
    """Evaluate one deterministic per-row mask against the same SVR metric."""
    y_tr = model.score(x_train)
    y_te = model.score(x_test)
    y_tr_keyed = keyed_masked_scores(model, x_train, train_keys, secret,
                                     alpha=1.0)
    y_te_keyed = keyed_masked_scores(model, x_test, test_keys, secret,
                                     alpha=1.0)
    svr = SVR(kernel="rbf", C=1.0, gamma="scale")
    svr.fit(x_train, y_tr_keyed)
    return {
        "r2_keyed_q1": float(r2_score(y_te, svr.predict(x_test))),
        "r2_keyed_repeated_q25": float(
            r2_score(y_te, svr.predict(x_test))),
        "note": "Q=25 repeats the same keyed response, so averaging is identical to Q=1",
        "signed_bias": float(np.mean(np.concatenate([y_tr_keyed - y_tr,
                                                      y_te_keyed - y_te]))),
        "mean_abs_perturbation": float(np.mean(np.abs(np.concatenate([
            y_tr_keyed - y_tr, y_te_keyed - y_te])))),
    }


def per_row_reconstruction(model, x_train, x_test, epsilon: float = 2.0,
                           alpha_max: float = 1.0, seed: int = 7,
                           noise_draws: int = 50,
                           model_scope: str = "unspecified") -> dict:
    """Referee M2: per-row / worst-case reconstruction error on concentrated rows.

    Buckets the in-window test rows into terciles of the dispersion ``S_t``
    (low ``S_t`` = concentrated firing) and reports, for every scheme:
      * realised per-row noise SD (mean over ``noise_draws`` masks);
      * the attacker's per-row ``|prediction - clean score|`` mean / p90 /
        max and per-bucket R^2 under the paper's in-window SVR protocol
        (fit on masked train labels, scored against clean test scores);
      * for adaptive, the share of cap-bound rows in the bucket.

    The fixed-matched comparison uses the fixed scale whose pooled noise SD
    is closest to the adaptive scheme's on this panel.
    """
    y_tr = model.score(x_train)
    y_te = model.score(x_test)
    firing_tr, f_tr = _firing(model, x_train)
    firing_te, f_te = _firing(model, x_test)

    def _dispersion(firing, f):
        total = firing.sum(axis=1, keepdims=True)
        w_bar = firing / np.where(total > 0, total, 1.0)
        y_raw = np.sum(w_bar * f, axis=1, keepdims=True)
        return np.sum(w_bar ** 2 * (f - y_raw) ** 2, axis=1)

    s_tr = _dispersion(firing_tr, f_tr)
    s_te = _dispersion(firing_te, f_te)
    keff_te = effective_rules(firing_te)
    bucket_idx = [np.asarray(b) for b in np.array_split(np.argsort(s_te), 3)]
    labels = ("low_S_concentrated", "mid_S", "high_S_dispersed")

    # Deployed single draw per scheme (attacker labels + per-row alphas).
    rng = np.random.default_rng(seed)
    y_tr_ad, a_tr, _ = adaptive_masked_firing_scores(
        model, x_train, epsilon=epsilon, alpha_max=alpha_max, rng=rng)
    y_te_ad, a_te, _ = adaptive_masked_firing_scores(
        model, x_test, epsilon=epsilon, alpha_max=alpha_max, rng=rng)
    ad_sd = float(np.concatenate([y_tr_ad - y_tr, y_te_ad - y_te]).std())

    best_alpha, best_gap, best_sd = 0.05, np.inf, 0.0
    for a in np.linspace(0.05, 1.0, 20):
        r = np.random.default_rng(seed)
        tr = masked_firing_scores(model, x_train, alpha=float(a), rng=r)
        te = masked_firing_scores(model, x_test, alpha=float(a), rng=r)
        sd = float(np.concatenate([tr - y_tr, te - y_te]).std())
        if abs(sd - ad_sd) < best_gap:
            best_alpha, best_gap, best_sd = float(a), abs(sd - ad_sd), sd
    alpha_matched = round(best_alpha, 2)

    obs = {
        "clean": y_tr,
        "fixed_005": masked_firing_scores(
            model, x_train, alpha=0.05, rng=np.random.default_rng(seed)),
        "fixed_matched": masked_firing_scores(
            model, x_train, alpha=alpha_matched, rng=np.random.default_rng(seed)),
        "adaptive": y_tr_ad,
    }
    preds, abs_err = {}, {}
    for name, y_obs in obs.items():
        est = SVR(kernel="rbf", C=1.0, gamma="scale")
        est.fit(x_train, y_obs)
        preds[name] = est.predict(x_test)
        abs_err[name] = np.abs(preds[name] - y_te)

    def _noise_sd(draw_fn) -> np.ndarray:
        outs = np.empty((noise_draws, len(x_test)))
        for d in range(noise_draws):
            outs[d] = draw_fn(np.random.default_rng(seed * 100003 + d))
        return outs.std(axis=0)

    noise_sd = {
        "fixed_005": _noise_sd(lambda r: masked_firing_scores(
            model, x_test, alpha=0.05, rng=r)),
        "fixed_matched": _noise_sd(lambda r: masked_firing_scores(
            model, x_test, alpha=alpha_matched, rng=r)),
        "adaptive": _noise_sd(lambda r: adaptive_masked_firing_scores(
            model, x_test, epsilon=epsilon, alpha_max=alpha_max, rng=r)[0]),
    }
    cap_bound = a_te >= alpha_max - 1e-12

    buckets = {}
    for label, idx in zip(labels, bucket_idx):
        entry = {
            "n": int(len(idx)),
            "s_t_median": float(np.median(s_te[idx])),
            "keff_median": float(np.median(keff_te[idx])),
            "adaptive_cap_rate": float(cap_bound[idx].mean()),
            "schemes": {},
        }
        for name in obs:
            entry["schemes"][name] = {
                "noise_sd_mean": 0.0 if name == "clean"
                else float(noise_sd[name][idx].mean()),
                "abs_err_mean": float(abs_err[name][idx].mean()),
                "abs_err_p90": float(np.quantile(abs_err[name][idx], 0.90)),
                "abs_err_max": float(abs_err[name][idx].max()),
                "r2": float(r2_score(y_te[idx], preds[name][idx])),
            }
        buckets[label] = entry
    return {
        "matched_alpha": alpha_matched,
        "adaptive_noise_sd_pooled": ad_sd,
        "matched_noise_sd_pooled": best_sd,
        "s_t_median_train": float(np.median(s_tr)),
        "buckets": buckets,
        "note": "test rows terciled by S_t (low = concentrated firing); "
                "attacker = paper's in-window SVR scored against clean test "
                "scores (single seeded draw); noise SD over "
                f"{noise_draws} independent masks; {model_scope} model",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="adaptive masking validation")
    ap.add_argument("--epsilon", type=float, default=2.0)
    ap.add_argument("--alpha-max", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--model", default=None)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    cfg = load_config()
    model_path = Path(args.model or Path(cfg["paths"]["models"]) / "panel_anfis.json")
    model = TSKFuzzySystem.load(model_path)
    train, test = _panel(cfg)
    x_train = train[list(FEATURE_COLUMNS)].to_numpy(dtype=float)
    x_test = test[list(FEATURE_COLUMNS)].to_numpy(dtype=float)
    train_keys = (train["ticker"].astype(str) + "\0"
                  + train["date"].astype(str)).to_numpy()
    test_keys = (test["ticker"].astype(str) + "\0"
                 + test["date"].astype(str)).to_numpy()

    report = {
        "model_path": str(model_path),
        "model_scope": ("frozen primary" if model_path.name == "panel_anfis_frozen.json"
                        else "deployed sensitivity"),
        "keff": dict(keff_diagnostic(model, x_test), scope="test pre-AGM"),
        "variance": variance_check(model, x_test, seed=args.seed),
        "adversary": adversary(model, x_train, x_test, args.epsilon,
                               args.alpha_max, args.seed),
        "keyed_adversary": keyed_adversary(
            model, x_train, x_test, train_keys, test_keys),
        "per_row": per_row_reconstruction(model, x_train, x_test, args.epsilon,
                                          args.alpha_max, args.seed,
                                          model_scope=("frozen primary"
                                                       if model_path.name == "panel_anfis_frozen.json"
                                                       else "deployed sensitivity")),
    }
    adv = report["adversary"]
    ad_noise = float(np.concatenate([
        adaptive_masked_firing_scores(
            model, x_train, epsilon=args.epsilon, alpha_max=args.alpha_max,
            rng=np.random.default_rng(args.seed))[0] - model.score(x_train),
        adaptive_masked_firing_scores(
            model, x_test, epsilon=args.epsilon, alpha_max=args.alpha_max,
            rng=np.random.default_rng(args.seed))[0] - model.score(x_test),
    ]).std())
    report["matched_noise"] = fixed_at_matched_noise(
        model, x_train, x_test, ad_noise, seed=args.seed)
    logs = Path(cfg["paths"]["logs"])
    logs.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output) if args.output else logs / "adaptive_masking.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
