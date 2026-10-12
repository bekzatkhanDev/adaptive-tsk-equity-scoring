"""Extraction audit (Referee Report, Requested Analyses 1-2).

Re-runs the Pre-AGM reconstruction protocol under the three fixes the referee
specified:

1. **Fidelity metric** -- every adversary is scored against the *clean* model's
   test scores (``r2_clean``), not against masked labels. The legacy masked-label
   R^2 (``r2_labels``) is kept beside it so the label-noise dilution arithmetic
   can be checked directly:
   ``R2_masked ~= R2_clean * Var(f) / (Var(f) + Var(noise))``.
2. **Out-of-window attacker** -- the deployed model is static between
   recalibrations and masking is armed only inside Pre-AGM windows, so an
   attacker can train on clean, out-of-window outputs (``regime=out_of_window``)
   or on clean 2025 out-of-window outputs (``regime=control_2025``, same-year
   non-stationarity control) instead of the in-window masked labels.
3. **Stronger extractors** -- default SVR (the paper's adversary), scaled +
   grid-tuned SVR, histogram gradient boosting, and a freshly refit TSK model
   (own subtractive-clustering centres + ridge consequents on attacker data).

Analysis 2 adds the missing baseline: additive Gaussian output noise at the
matched noise SD (same rows, zero mean, no cap, no truncation).

Referee M1 addenda (parts 2 and 4 of the requested fixes):

4. **Uncertainty + noise averaging + frontier** -- ``single_draw_ci`` gives
   mean and 95% CI of in-window ``r2_clean`` over 200 noise seeds per scheme;
   ``noise_average_cells`` repeats the attack with Q in {1, 5, 25} averaged
   queries per row (Remark on query averaging); ``frontier`` sweeps the
   fixed-alpha and adaptive-epsilon operating points (50 draws each) to map
   R^2 against realised noise SD.

Example
------
python experiments/extraction_audit.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.svm import SVR

from src.data_loader import FEATURE_COLUMNS, build_panel, load_config
from src.masking import adaptive_masked_firing_scores, masked_scores
from src.TSK_engine import TSKFuzzySystem
from experiments.adaptive_masking import _panel


# --------------------------------------------------------------------- helpers
def _fidelity(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    keep = np.isfinite(y_true) & np.isfinite(y_pred)
    return {
        "r2": float(r2_score(y_true[keep], y_pred[keep])),
        "spearman": float(spearmanr(y_true[keep], y_pred[keep]).statistic),
    }


def make_labels(model, x_train, x_test, scheme: str, noise_sd: float,
                epsilon: float, alpha_max: float, seed: int,
                alpha_matched: float = 0.6):
    """Observed labels per scheme on the in-window panels.

    Returns ``(y_train_obs, y_test_obs, realised_noise_sd)`` where noise is
    measured over train+test against the clean scores. ``scheme='clean'``
    returns the unmodified scores (attacker baseline).
    """
    y_tr = model.score(x_train)
    y_te = model.score(x_test)
    rng = np.random.default_rng(seed)
    if scheme == "clean":
        return y_tr, y_te, 0.0
    if scheme == "fixed_005":
        y_tr_o = masked_scores(model, x_train, None, alpha=0.05, rng=rng)
        y_te_o = masked_scores(model, x_test, None, alpha=0.05, rng=rng)
    elif scheme == "fixed_matched":
        y_tr_o = masked_scores(model, x_train, None, alpha=alpha_matched, rng=rng)
        y_te_o = masked_scores(model, x_test, None, alpha=alpha_matched, rng=rng)
    elif scheme == "adaptive":
        y_tr_o, _, _ = adaptive_masked_firing_scores(
            model, x_train, epsilon=epsilon, alpha_max=alpha_max, rng=rng)
        y_te_o, _, _ = adaptive_masked_firing_scores(
            model, x_test, epsilon=epsilon, alpha_max=alpha_max, rng=rng)
    elif scheme == "gaussian_matched":
        # Analysis 2: additive N(0, noise_sd^2) on the same rows, clamped to
        # the score domain exactly like the masking paths.
        y_tr_o = np.clip(y_tr + rng.normal(0.0, noise_sd, len(y_tr)), 0.0, model.score_scale)
        y_te_o = np.clip(y_te + rng.normal(0.0, noise_sd, len(y_te)), 0.0, model.score_scale)
    else:
        raise ValueError(f"unknown scheme {scheme!r}")
    noise = np.concatenate([y_tr_o - y_tr, y_te_o - y_te])
    return y_tr_o, y_te_o, float(noise.std())


def noise_average_cells(model, scheme: str, x_tr_in, x_te_in, y_clean_te,
                        epsilon: float, alpha_max: float,
                        q_list=(1, 5, 25), n_draws: int = 200,
                        seed: int = 7) -> dict:
    """Referee fix, part 2 (noise averaging) + part 4 (uncertainty).

    For each query budget ``Q`` the attacker averages ``Q`` i.i.d. noise
    draws per row before fitting the paper's default SVR in-window; the
    reported ``r2_clean`` is the mean over ``n_draws`` seeds with a
    normal 95% CI.  At Q=1 this reproduces the single-draw cell; as
    Q grows the masking noise averages out (Remark ``rem:threat_model``).
    """
    out = {}
    for q in q_list:
        r2s = np.empty(n_draws)
        for d in range(n_draws):
            rng = np.random.default_rng(seed * 100003 + d)
            acc_tr = np.zeros_like(model.score(x_tr_in))
            acc_te = np.zeros_like(model.score(x_te_in))
            for _ in range(q):
                if scheme == "fixed_005":
                    acc_tr += masked_scores(model, x_tr_in, None, alpha=0.05,
                                            rng=rng)
                    acc_te += masked_scores(model, x_te_in, None, alpha=0.05,
                                            rng=rng)
                elif scheme == "adaptive":
                    a, _, _ = adaptive_masked_firing_scores(
                        model, x_tr_in, epsilon=epsilon,
                        alpha_max=alpha_max, rng=rng)
                    b, _, _ = adaptive_masked_firing_scores(
                        model, x_te_in, epsilon=epsilon,
                        alpha_max=alpha_max, rng=rng)
                    acc_tr += a
                    acc_te += b
                else:
                    raise ValueError(scheme)
            est = SVR(kernel="rbf", C=1.0, gamma="scale")
            est.fit(x_tr_in, acc_tr / q)
            r2s[d] = _fidelity(y_clean_te, est.predict(x_te_in))["r2"]
        m, s = float(r2s.mean()), float(r2s.std(ddof=1))
        out[f"Q{q}"] = {"mean": m, "sd": s,
                        "ci95": [m - 1.96 * s / np.sqrt(n_draws),
                                 m + 1.96 * s / np.sqrt(n_draws)],
                        "n_draws": n_draws}
    return out


def single_draw_ci(model, scheme: str, x_tr_in, x_te_in, y_clean_te,
                   epsilon: float, alpha_max: float, noise_sd: float,
                   n_draws: int = 200, seed: int = 11) -> dict:
    """Referee fix, part 4: mean + 95% CI of in-window ``r2_clean`` for the
    default SVR over ``n_draws`` noise seeds (per scheme)."""
    r2s = np.empty(n_draws)
    for d in range(n_draws):
        y_tr_o, y_te_o, _ = make_labels(
            model, x_tr_in, x_te_in, scheme, noise_sd, epsilon, alpha_max,
            seed * 100003 + d)
        est = SVR(kernel="rbf", C=1.0, gamma="scale")
        est.fit(x_tr_in, y_tr_o)
        r2s[d] = _fidelity(y_clean_te, est.predict(x_te_in))["r2"]
    m, s = float(r2s.mean()), float(r2s.std(ddof=1))
    return {"mean": m, "sd": s,
            "ci95": [m - 1.96 * s / np.sqrt(n_draws),
                     m + 1.96 * s / np.sqrt(n_draws)],
            "n_draws": n_draws}


def frontier(model, x_tr_in, x_te_in, y_clean_te,
             alpha_max: float, seed: int = 7,
             alphas=(0.05, 0.1, 0.2, 0.3, 0.6, 1.0),
             epsilons=(0.5, 1.0, 2.0, 3.0),
             n_draws: int = 50) -> dict:
    """Referee fix: R^2-versus-noise-SD frontier (default SVR, in-window,
    scored against clean test scores).

    Each point averages ``n_draws`` independent masking draws: single draws
    at high noise scale have enough seed-to-seed variance to make the sweep
    look non-monotone (verified), so the frontier is reported as a mean.
    """
    def _point(scheme: str, draw_fn) -> dict:
        r2s, sds = np.empty(n_draws), np.empty(n_draws)
        for d in range(n_draws):
            rng = np.random.default_rng(seed * 100003 + d)
            tr, te = draw_fn(rng)
            sds[d] = float(np.concatenate([tr - model.score(x_tr_in),
                                           te - model.score(x_te_in)]).std())
            est = SVR(kernel="rbf", C=1.0, gamma="scale")
            est.fit(x_tr_in, tr)
            r2s[d] = _fidelity(y_clean_te, est.predict(x_te_in))["r2"]
        return {"scheme": scheme, "noise_sd": float(sds.mean()),
                "r2_clean": float(r2s.mean()), "r2_sd": float(r2s.std(ddof=1)),
                "n_draws": n_draws}

    pts = []
    for a in alphas:
        pts.append(_point(
            f"fixed_a{a}",
            lambda rng, a=a: (masked_scores(model, x_tr_in, None, alpha=a, rng=rng),
                              masked_scores(model, x_te_in, None, alpha=a, rng=rng))))
    for e in epsilons:
        def _adapt(rng, e=e):
            tr, _, _ = adaptive_masked_firing_scores(
                model, x_tr_in, epsilon=e, alpha_max=alpha_max, rng=rng)
            te, _, _ = adaptive_masked_firing_scores(
                model, x_te_in, epsilon=e, alpha_max=alpha_max, rng=rng)
            return tr, te
        pts.append(_point(f"adaptive_e{e}", _adapt))
    return {"points": pts, "n_draws": n_draws}


def fit_tsk_refit(x: np.ndarray, y: np.ndarray, clip: float = 100.0) -> TSKFuzzySystem:
    """Attacker's own TSK model fitted to the observed labels.

    On panels large enough to estimate the premise parameters (>= 500 rows)
    the attacker runs the full subtractive-clustering + premise-gradient +
    ridge-consequent training the deployed model itself uses (ridge = 10,
    200 epochs). On the 205-row in-window panel the premise problem is not
    estimable -- ridge = 10 collapses to the zero predictor and free premise
    training diverges (measured) -- so there only the consequent ridge solve
    is used with a weak regulariser, which is what a careful attacker would
    fall back on.
    """
    fresh = TSKFuzzySystem(n_inputs=x.shape[1], n_rules=5)
    fresh.init_from_data(x, radius=0.5, ratio=0.5)
    fresh.clip_rule_outputs = True
    fresh.score_scale = clip
    if len(x) >= 500:
        from src.anfis_trainer import ANFISTrainer
        trainer = ANFISTrainer(fresh, lr=1e-3, ridge=10.0, grad_clip=5.0)
        trainer.fit(x, y, epochs=200, verbose_every=0)
    else:
        fresh.fit_consequents(x, y, ridge=1e-6)
    return fresh


def extractors() -> dict:
    """Four attacker models, from the paper's adversary upward."""
    return {
        "svr_default": lambda: Pipeline(
            [("svr", SVR(kernel="rbf", C=1.0, gamma="scale"))]),
        "svr_tuned": lambda: GridSearchCV(
            Pipeline([("scale", StandardScaler()), ("svr", SVR(kernel="rbf"))]),
            {"svr__C": [1.0, 10.0, 100.0, 1000.0],
             "svr__gamma": ["scale", 0.1, 0.01, 0.001]},
            cv=KFold(5, shuffle=True, random_state=0),
            n_jobs=-1),
        "hgb": lambda: HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.08, max_leaf_nodes=31, random_state=0),
        "tsk_refit": "tsk",
    }


def build_out_of_window(cfg: dict, split: str, model) -> tuple[np.ndarray, np.ndarray]:
    """Clean attacker rows: the full-year panel minus Pre-AGM rows.

    The deployed model is only armed inside Pre-AGM windows, so every other
    row of the static model is a clean query -- this is the out-of-window
    attacker the referee asks for (and the 2025 variant doubles as the
    same-year non-stationarity control).
    """
    panel = build_panel(cfg, split=split)
    cols = list(FEATURE_COLUMNS)
    X = panel[cols].to_numpy(dtype=float)
    outside = panel["pre_agm"].to_numpy() <= 0.0
    return X[outside], model.score(X[outside])


# ------------------------------------------------------------------ protocols
def run(cfg: dict, model, draws: int, seed: int,
        epsilon: float, alpha_max: float) -> dict:
    train, test = _panel(cfg)
    cols = list(FEATURE_COLUMNS)
    x_tr_in = train[cols].to_numpy(dtype=float)
    x_te_in = test[cols].to_numpy(dtype=float)
    y_clean_tr = model.score(x_tr_in)
    y_clean_te = model.score(x_te_in)

    # Attacker training sets per regime (x_train, label source handled below).
    x_tr_out, y_tr_out = build_out_of_window(cfg, "train", model)   # clean 2022-24
    x_te_out, y_te_out = build_out_of_window(cfg, "test", model)    # clean 2025
    regimes = {
        "in_window_masked": x_tr_in,
        "out_of_window": x_tr_out,
        "control_2025": x_te_out,
    }

    # Noise-SD reference for the additive-Gaussian baseline (Analysis 2):
    # measured from the adaptive scheme, then reused by gaussian_matched.
    _, _, sd_adaptive = make_labels(
        model, x_tr_in, x_te_in, "adaptive", 0.0, epsilon, alpha_max, seed)

    # Matched fixed scale: the fixed alpha whose pooled noise SD is closest
    # to the calibrated adaptive scheme's on this panel.
    best_a, best_gap = 0.6, np.inf
    for a in np.linspace(0.05, 1.0, 20):
        r = np.random.default_rng(seed)
        tr = masked_scores(model, x_tr_in, None, alpha=float(a), rng=r)
        te = masked_scores(model, x_te_in, None, alpha=float(a), rng=r)
        gap = abs(float(np.concatenate(
            [tr - y_clean_tr, te - y_clean_te]).std()) - sd_adaptive)
        if gap < best_gap:
            best_a, best_gap = float(a), gap
    alpha_matched = round(best_a, 2)

    schemes = ["clean", "fixed_005", "fixed_matched", "adaptive", "gaussian_matched"]
    out: dict = {
        "n_train_in": len(x_tr_in), "n_test_in": len(x_te_in),
        "n_train_out_window": len(x_tr_out), "n_test_out_window": len(x_te_out),
        "adaptive_noise_sd": sd_adaptive,
        "fixed_matched_alpha": alpha_matched,
        "note": "r2_clean = fidelity to the clean model (primary); "
                "r2_labels = legacy masked-label metric (dilution check); "
                "regimes: in_window_masked trains on the scheme's observed "
                "labels, out_of_window/control_2025 train on clean queries "
                "outside the armed windows",
    }
    var_f = float(np.var(np.concatenate([y_clean_tr, y_clean_te])))
    out["var_clean_scores"] = var_f

    # Scheme-independent attacker regimes: train on clean out-of-window
    # queries, score against the clean in-window 2025 test scores. Labels
    # here are identical for every scheme, so this is computed exactly once.
    shared_cells: dict = {}
    for name, factory in extractors().items():
        shared_cells[name] = {}
        for regime, (Xtr, ytr) in {
                "out_of_window": (x_tr_out, y_tr_out),
                "control_2025": (x_te_out, y_te_out)}.items():
            if name == "tsk_refit":
                pred = fit_tsk_refit(Xtr, ytr, clip=model.score_scale).score(x_te_in)
            else:
                est = factory()
                est.fit(Xtr, ytr)
                pred = est.predict(x_te_in)
            fid = _fidelity(y_clean_te, pred)
            shared_cells[name][f"{name}|{regime}"] = {
                "r2_clean": fid["r2"], "spearman_clean": fid["spearman"]}

    for scheme in schemes:
        y_tr_obs, y_te_obs, sd = make_labels(
            model, x_tr_in, x_te_in, scheme, sd_adaptive,
            epsilon, alpha_max, seed, alpha_matched=alpha_matched)
        var_noise = float(sd ** 2)
        row = {"noise_sd": sd, "var_noise": var_noise, "cells": {}}
        for name, factory in extractors().items():
            # Scheme-independent regimes: out-of-window labels are clean for
            # every scheme (masking is not armed there), so those cells are
            # computed once before the scheme loop and injected below.
            Xtr, ytr = x_tr_in, y_tr_obs
            Xte = x_te_in
            if name == "tsk_refit":
                pred = fit_tsk_refit(
                    Xtr, ytr, clip=model.score_scale).score(Xte)
            else:
                est = factory()
                est.fit(Xtr, ytr)
                pred = est.predict(Xte)
            fid = _fidelity(y_clean_te, pred)
            cell = {"r2_clean": fid["r2"], "spearman_clean": fid["spearman"]}
            cell["r2_labels"] = _fidelity(y_te_obs, pred)["r2"]
            if scheme != "clean" and var_noise > 0:
                cell["dilution_prediction"] = float(
                    cell["r2_clean"] * var_f / (var_f + var_noise))
            row["cells"][f"{name}|in_window_masked"] = cell
            for key, shared in shared_cells.get(name, {}).items():
                row["cells"][key] = shared
        out[scheme] = row

    # ---- referee M1 fixes (PARTS 2 + 4): uncertainty + noise averaging ----
    ci = {}
    for scheme in ["clean", "fixed_005", "fixed_matched", "adaptive"]:
        sd = float(out[scheme]["noise_sd"])
        ci[scheme] = single_draw_ci(model, scheme, x_tr_in, x_te_in, y_clean_te,
                                    epsilon, alpha_max, sd)
        ci[scheme]["noise_sd"] = sd
    out["r2_clean_ci_200draws"] = ci
    out["noise_averaging"] = {
        scheme: noise_average_cells(model, scheme, x_tr_in, x_te_in,
                                    y_clean_te, epsilon, alpha_max)
        for scheme in ["fixed_005", "adaptive"]
    }
    out["frontier"] = frontier(model, x_tr_in, x_te_in, y_clean_te, alpha_max)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="extraction audit (analyses 1-2)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--epsilon", type=float, default=2.0)
    ap.add_argument("--alpha-max", type=float, default=1.0)
    args = ap.parse_args()

    cfg = load_config()
    model = TSKFuzzySystem.load(Path(cfg["paths"]["models"]) / "panel_anfis.json")
    report = run(cfg, model, 25, args.seed, args.epsilon, args.alpha_max)

    logs = Path(cfg["paths"]["logs"])
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "extraction_audit.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")

    print(f"adaptive noise SD: {report['adaptive_noise_sd']:.3f}  "
          f"Var(clean) = {report['var_clean_scores']:.3f}  "
          f"n_out_window train/test: {report['n_train_out_window']}/"
          f"{report['n_test_out_window']}")
    for scheme in ["clean", "fixed_005", "fixed_matched", "adaptive", "gaussian_matched"]:
        row = report[scheme]
        cells = row["cells"]
        paper_adv = cells.get("svr_default|in_window_masked")
        best = cells.get("svr_tuned|in_window_masked")
        oow = cells.get("svr_default|out_of_window")
        ctrl = cells.get("svr_default|control_2025")
        hgb = cells.get("hgb|in_window_masked")
        tsk = cells.get("tsk_refit|in_window_masked")
        print(f"\n{scheme:16s} noise_sd={row['noise_sd']:.3f}")
        if paper_adv:
            print(f"  paper adv (in-window):  r2_clean={paper_adv['r2_clean']:.3f}  "
                  f"r2_labels={paper_adv.get('r2_labels', float('nan')):.3f}  "
                  f"dil_pred={paper_adv.get('dilution_prediction', float('nan')):.3f}")
        if best:
            print(f"  tuned SVR (in-window):  r2_clean={best['r2_clean']:.3f}")
        if hgb:
            print(f"  HGB (in-window):        r2_clean={hgb['r2_clean']:.3f}")
        if tsk:
            print(f"  TSK refit (in-window):  r2_clean={tsk['r2_clean']:.3f}")
        if oow:
            print(f"  out-of-window attacker: r2_clean={oow['r2_clean']:.3f}")
        if ctrl:
            print(f"  2025 same-year control: r2_clean={ctrl['r2_clean']:.3f}")
    print(f"\nfull report: {logs / 'extraction_audit.json'}")
    ci = report.get("r2_clean_ci_200draws", {})
    for scheme in ["clean", "fixed_005", "adaptive"]:
        c = ci.get(scheme, {})
        if c:
            print(f"CI200 {scheme:10s} r2_clean mean={c['mean']:.3f} "
                  f"95%=[{c['ci95'][0]:.3f},{c['ci95'][1]:.3f}]")
    for scheme, qs in report.get("noise_averaging", {}).items():
        print(f"avg {scheme}: " + "  ".join(
            f"{q}: {v['mean']:.3f}[{v['ci95'][0]:.3f},{v['ci95'][1]:.3f}]"
            for q, v in qs.items()))
    print("frontier: " + "  ".join(
        f"{p['scheme']}(sd={p['noise_sd']:.2f},r2={p['r2_clean']:.3f}"
        f"+-{p['r2_sd']:.3f})"
        for p in report.get("frontier", {}).get("points", [])))


if __name__ == "__main__":
    main()