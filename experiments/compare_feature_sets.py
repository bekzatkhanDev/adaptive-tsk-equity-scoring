"""Compare feature sets for the TSK scorer (Tier-2 evaluation).

Trains the same ANFIS protocol (2024 train -> 2025 out-of-sample) on several
input vectors and reports pooled Rank IC, RMSE/MAE against the Eq. (10) target,
and the net Sharpe of the monthly top-N portfolio, so the marginal value of the
Tier-2 enrichment is measured rather than asserted:

* ``baseline``              -- the canonical three Section-3.1 inputs;
* ``microstructure``        -- baseline + multi-scale trailing OHLC features;
* ``fundamentals``          -- baseline + as-of (point-in-time) slow fundamentals;
* ``full``                  -- baseline + every Tier-2 column.

Tier-2 vectors are min-max scaled inside the model (``input_scaling: minmax``);
the baseline keeps the canonical ``raw`` path, so the baseline row reproduces
``train_anfis_panel.py`` exactly.

Because the sample carries only seven assets over a single out-of-sample year,
the reported gain of one feature set could still be an artefact of the exact
training draw. With ``--seeds N`` each set is retrained N times on i.i.d. row
bootstraps of the 2024 training panel (seed = ``project.seed`` shifted by
0..N-1; the test panel is never resampled) and the table reports mean +/- SE
per metric plus the per-set seed spread -- a bootstrap-stability check of the
ranking, not a claim about market-time variation.

Example
-------
python experiments/compare_feature_sets.py
python experiments/compare_feature_sets.py --seeds 5   # robustness check
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.anfis_trainer import ANFISTrainer
from src.backtest import (deflated_sharpe_ratio, mark_rank_ic, net_sharpe,
                          pooled_rank_ic, top_n_portfolio_returns,
                          top_n_spread_returns)
from src.data_loader import build_panel, feature_columns, load_config
from src.TSK_engine import TSKFuzzySystem

# Default run == the previously published comparison (single seed, cfg seed).
DEFAULT_SETS = {
    "baseline": {"extra_features": [], "fundamentals": False, "input_scaling": "raw"},
    "microstructure": {
        "extra_features": ["dev_price_60", "ret_5", "vol_20", "clv_5", "amihud"],
        "fundamentals": False,
        "input_scaling": "minmax",
    },
    "fundamentals": {
        "extra_features": [],
        "fundamentals": True,
        "input_scaling": "minmax",
    },
    "full": {
        "extra_features": ["dev_price_60", "ret_5", "vol_20", "clv_5", "amihud"],
        "fundamentals": True,
        "input_scaling": "minmax",
    },
}

# Ablation ladder: peel one microstructure input off the winning vector at a
# time (--ablations) to see whether its edge survives dropping any single
# column (a one-feature-dependent gain would collapse under leave-one-out).
ABLATION_SETS = {
    f"micro_minus_{name}": {
        "extra_features": [c for c in DEFAULT_SETS["microstructure"]["extra_features"]
                           if c != name],
        "fundamentals": False,
        "input_scaling": "minmax",
    }
    for name in DEFAULT_SETS["microstructure"]["extra_features"]
}

METRIC_COLUMNS = ("pooled_rank_ic", "rmse", "mae", "net_sharpe_15bps",
                  "spread_sharpe_15bps", "monthly_mark_ic", "quarterly_mark_ic")


def _seeded_cfg(cfg: dict, seed: int) -> dict:
    """Copy of ``cfg`` whose randomness is driven by ``seed``.

    ``project.seed`` is set for bookkeeping; the actual stochastic draw is a
    row bootstrap of the *training* panel (the only place RNG can enter the
    pipeline -- clustering, LS and GD are all deterministic). Test rows are
    never resampled, so out-of-sample evaluation stays exact.
    """
    seeded = copy.deepcopy(cfg)
    seeded.setdefault("project", {})["seed"] = int(seed)
    seeded.setdefault("features", {})["train_bootstrap"] = True
    return seeded


def fit_variant(cfg: dict, spec: dict, epochs: int):
    """Train one feature-set variant and score the 2025 out-of-sample panel.

    ``cfg["features"]["train_bootstrap"]`` (set only by ``--seeds N``) replaces
    the training matrix with an i.i.d. row bootstrap drawn from
    ``project.seed`` *after* the matrix is assembled -- clustering, the LS
    solve and GD all then see the resampled rows, so different seeds give
    genuinely different models. The test panel is never touched.
    """
    variant = json.loads(json.dumps(cfg))  # deep copy
    # Preserve any in-memory-only flags carried by ``cfg`` (e.g. the
    # ``train_bootstrap`` switch set by ``_seeded_cfg`` for --seeds runs);
    # ``config.yaml`` never contains them, so dropping them here would make
    # every seed train on identical rows and collapse the SE to zero.
    variant["features"].update({
        "extra_features": spec["extra_features"],
        "fundamentals": spec["fundamentals"],
        **{k: v for k, v in cfg.get("features", {}).items()
           if k not in ("extra_features", "fundamentals")},
    })
    variant["tsk"]["input_scaling"] = spec["input_scaling"]

    columns = list(feature_columns(variant))
    train = build_panel(variant, split="train").dropna(subset=["Y_target"])
    test = build_panel(variant, split="test")

    X = train[columns].to_numpy(dtype=float)
    y = train["Y_target"].to_numpy(dtype=float)
    if bool(variant["features"].get("train_bootstrap", False)):
        rng = np.random.default_rng(int(variant.get("project", {}).get("seed", 0)))
        idx = rng.integers(0, len(X), size=len(X))
        X, y = X[idx], y[idx]
    tsk = variant["tsk"]
    early_stopping = bool(spec.get("early_stopping", False))
    model = TSKFuzzySystem(
        n_inputs=len(columns),
        n_rules=int(tsk["n_rules"]),
        sigma_floor=float(variant["target"]["sigma_floor"]),
        firing_floor=float(variant["target"]["firing_floor"]),
        neutral_score=float(variant["target"]["neutral_score"]),
        score_scale=float(variant["target"]["score_scale"]),
        clip_rule_outputs=bool(tsk.get("clip_rule_outputs", True)),
        input_scaling=spec["input_scaling"],
        feature_names=columns,
    )
    model.init_from_data(X, radius=float(tsk["cluster_radius"]), ratio=float(tsk["cluster_ratio"]))
    trainer = ANFISTrainer(model, lr=float(tsk["premise_lr"]), ridge=float(tsk["ridge_lambda"]),
                           grad_clip=float(tsk["grad_clip"]))
    trainer.fit(X, y, epochs=int(epochs), verbose_every=0,
                early_stopping=early_stopping)

    scored = test.assign(Y_attr=model.score(test[columns].to_numpy(dtype=float)))
    labelled = scored.dropna(subset=["Y_target"])
    prediction = labelled["Y_attr"].to_numpy(dtype=float)
    target = labelled["Y_target"].to_numpy(dtype=float)
    portfolio = top_n_portfolio_returns(scored, top_n=int(cfg["backtest"]["top_n"]))
    spread = top_n_spread_returns(scored, top_n=int(cfg["backtest"]["top_n"]))
    monthly_ic = mark_rank_ic(scored, freq="monthly", horizon=21)[0]
    quarterly_ic = mark_rank_ic(scored, freq="quarterly", horizon=63)[0]
    cost = cfg["backtest"]["cost_tiers_bps"][0]
    return model, scored, {
        "n_inputs": len(columns),
        "features": ",".join(columns),
        "n_free_params": int(model.n_free_params),
        "epochs_run": int(model.training_metadata.get("n_epochs", epochs)),
        "rmse": float(np.sqrt(np.mean((prediction - target) ** 2))),
        "mae": float(np.mean(np.abs(prediction - target))),
        "pooled_rank_ic": pooled_rank_ic(scored),
        "net_sharpe_15bps": net_sharpe(portfolio, cost_bps=cost),
        "spread_sharpe_15bps": net_sharpe(spread, cost_bps=cost),
        "monthly_mark_ic": monthly_ic,
        "quarterly_mark_ic": quarterly_ic,
    }


def _net_spread_excess(panel: pd.DataFrame, cfg: dict) -> np.ndarray:
    """Cost-adjusted monthly excess returns of the long-short spread leg.

    Mirrors ``net_sharpe``'s turnover-based cost model (round-trip ``cost_bps``
    charged on the long-side membership churn) but returns the raw monthly
    excess series so it can feed ``deflated_sharpe_ratio``, which needs the
    per-period returns (not the annualised scalar) to estimate skew/kurtosis.
    """
    spread_df = top_n_spread_returns(panel, top_n=int(cfg["backtest"]["top_n"]))
    if spread_df.empty:
        return np.array([], dtype=float)
    cost = cfg["backtest"]["cost_tiers_bps"][0]
    excess = spread_df["portfolio_return"].to_numpy(dtype=float)
    membership = [set(p.split("|")[0].split(",")) for p in spread_df["picks"]]
    changes = [len(membership[i] ^ membership[i - 1]) / max(len(membership[i]), 1)
               for i in range(1, len(membership))]
    monthly_turnover = float(np.mean(changes)) / 2.0 if changes else 0.0
    return excess - monthly_turnover * cost / 1e4


def walk_forward(cfg: dict, spec: dict, epochs: int, n_folds: int) -> list[dict]:
    """Rolling retraining folds -- the honest small-sample evaluation protocol.

    The canonical single split yields only ~11 labelled spread months (T=11 is
    why the DSR cannot clear 95% here). Walk-forward instead cuts the panel
    into ``n_folds`` contiguous chronological test blocks; each fold refits
    scaling/centres/consequents/GD from scratch on every row strictly before
    its test block (train_start .. fold start) and scores only that block. The
    pooled out-of-sample months across folds multiply the effective evaluation
    sample without ever leaking future data into training.

    Returns one metrics record per fold plus a ``pooled_all_folds`` summary.
    """
    variant = json.loads(json.dumps(cfg))  # deep copy
    variant["features"].update({
        "extra_features": spec["extra_features"],
        "fundamentals": spec["fundamentals"],
        **{k: v for k, v in cfg.get("features", {}).items()
           if k not in ("extra_features", "fundamentals")},
    })
    variant["tsk"]["input_scaling"] = spec["input_scaling"]
    columns = list(feature_columns(variant))

    full = build_panel(variant, split="all").dropna(subset=["Y_target"])
    tsk = variant["tsk"]

    def _train_model(X, y):
        model = TSKFuzzySystem(
            n_inputs=len(columns),
            n_rules=int(tsk["n_rules"]),
            sigma_floor=float(variant["target"]["sigma_floor"]),
            firing_floor=float(variant["target"]["firing_floor"]),
            neutral_score=float(variant["target"]["neutral_score"]),
            score_scale=float(variant["target"]["score_scale"]),
            clip_rule_outputs=bool(tsk.get("clip_rule_outputs", True)),
            input_scaling=spec["input_scaling"],
            feature_names=columns,
        )
        model.init_from_data(X, radius=float(tsk["cluster_radius"]),
                             ratio=float(tsk["cluster_ratio"]))
        trainer = ANFISTrainer(model, lr=float(tsk["premise_lr"]),
                               ridge=float(tsk["ridge_lambda"]),
                               grad_clip=float(tsk["grad_clip"]))
        trainer.fit(X, y, epochs=int(epochs), verbose_every=0,
                    early_stopping=bool(spec.get("early_stopping", False)))
        return model

    # Test blocks: equal date ranges covering the *labelled* span. The last
    # ~30 calendar days of the panel never carry a realised R30 (the forward
    # window runs past the data end), so the walk-forward horizon is truncated
    # at the latest labelled first-of-month to avoid empty folds; every fold's
    # training set ends where its own test block begins.
    test_lo = pd.Timestamp(cfg["experiment"]["test_start"])
    test_hi = pd.Timestamp(cfg["experiment"]["test_end"])
    labelled_months = [g["date"].min() for _, g in
                       full[(full["date"] >= test_lo) & full["R30"].notna()]
                       .groupby(pd.Grouper(key="date", freq="MS"))]
    if labelled_months:
        test_hi = min(test_hi, max(labelled_months))
    edges = pd.date_range(test_lo, test_hi + pd.Timedelta(days=2), periods=n_folds + 1)
    records = []
    scored_parts = []
    for f in range(n_folds):
        fit_start = pd.Timestamp(cfg["experiment"]["train_start"])
        fit_end = edges[f]
        te_start, te_end = edges[f], edges[f + 1]
        train = full[(full["date"] >= fit_start) & (full["date"] < fit_end)]
        test = full[(full["date"] >= te_start) & (full["date"] < te_end)]
        if len(train) < 60 or len(test) < 10:
            continue
        model = _train_model(train[columns].to_numpy(float),
                             train["Y_target"].to_numpy(float))
        scored = test.assign(Y_attr=model.score(test[columns].to_numpy(float)))
        scored_parts.append(scored)
        portfolio = top_n_portfolio_returns(scored, top_n=int(cfg["backtest"]["top_n"]))
        spread = top_n_spread_returns(scored, top_n=int(cfg["backtest"]["top_n"]))
        records.append({
            "fold": f,
            "fit_end": str(fit_end.date()),
            "test_start": str(te_start.date()),
            "test_end": str((te_end - pd.Timedelta(days=1)).date()),
            "n_train_rows": int(len(train)),
            "pooled_rank_ic": pooled_rank_ic(scored),
            "net_sharpe_15bps": net_sharpe(portfolio,
                                           cost_bps=cfg["backtest"]["cost_tiers_bps"][0])
            if len(portfolio) > 1 else float("nan"),
            "spread_sharpe_15bps": net_sharpe(spread,
                                              cost_bps=cfg["backtest"]["cost_tiers_bps"][0])
            if len(spread) > 1 else float("nan"),
            "_spread": spread,
        })
    pooled_panel = pd.concat(scored_parts, ignore_index=True)
    pooled_spread = top_n_spread_returns(pooled_panel, top_n=int(cfg["backtest"]["top_n"]))
    pooled_portfolio = top_n_portfolio_returns(pooled_panel, top_n=int(cfg["backtest"]["top_n"]))
    pooled = {
        "fold": "pooled_all_folds",
        "fit_end": "", "test_start": str(test_lo.date()), "test_end": str(test_hi.date()),
        "n_train_rows": int(len(full[full["date"] < test_lo])),
        "pooled_rank_ic": pooled_rank_ic(pooled_panel),
        "net_sharpe_15bps": net_sharpe(pooled_portfolio,
                                       cost_bps=cfg["backtest"]["cost_tiers_bps"][0]),
        "spread_sharpe_15bps": net_sharpe(pooled_spread,
                                          cost_bps=cfg["backtest"]["cost_tiers_bps"][0]),
        "_spread": pooled_spread,
    }
    return records + [pooled], pooled_panel


def _run_walk_forward(cfg: dict, sets: dict, epochs: int, n_folds: int) -> None:
    """Walk-forward driver: per-set fold tables + pooled comparison + DSR."""
    all_fold_rows, pooled_records = [], []
    for name, spec in sets.items():
        records, pooled_panel = walk_forward(cfg, spec, epochs, n_folds)
        for r in records:
            spread_df = r.pop("_spread")
            r["n_spread_months"] = 0 if spread_df is None or spread_df.empty \
                else int(len(spread_df))
            if r["fold"] == "pooled_all_folds":
                excess = _net_spread_excess(pooled_panel, cfg)
                r["dsr_pooled"] = deflated_sharpe_ratio(
                    excess, n_trials=len(sets), periodicity=12)["dsr"]
                pooled_records.append({**r, "set": name})
            else:
                all_fold_rows.append({**r, "set": name})
        print(f"{name:20} folds done; pooled IC="
              f"{pooled_records[-1]['pooled_rank_ic']:+.4f} "
              f"spread Sharpe={pooled_records[-1]['spread_sharpe_15bps']:+.3f}")

    pooled_table = pd.DataFrame(pooled_records)[
        ["set", "n_train_rows", "n_spread_months", "pooled_rank_ic",
         "net_sharpe_15bps", "spread_sharpe_15bps", "dsr_pooled"]]
    if len(pooled_table):
        winner = pooled_table.loc[pooled_table["pooled_rank_ic"].idxmax(), "set"]
        # Honest DSR: n_trials = number of feature sets actually compared.
        win_mask = pooled_table["set"] == winner
        # recompute the winner's pooled excess with its own trial count
        _, win_panel = walk_forward(cfg, sets[winner], epochs, n_folds)
        excess = _net_spread_excess(win_panel, cfg)
        dsr = deflated_sharpe_ratio(excess, n_trials=len(sets), periodicity=12)
        pooled_table.loc[win_mask, "dsr_pooled"] = dsr["dsr"]
        print(f"\nWalk-forward winner: {winner}  "
              f"DSR(n_trials={len(sets)})={dsr['dsr']:.3f}  "
              f"SR_obs={dsr['sharpe_obs']:+.3f}  SR*={dsr['sr_star']:+.3f}  "
              f"T={len(excess)} months")

    exports = Path(cfg["paths"]["csv_exports"])
    logs = Path(cfg["paths"]["logs"])
    exports.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    stem = f"walk_forward_{n_folds}fold"
    pooled_table.to_csv(exports / f"{stem}_pooled.csv", index=False)
    pd.DataFrame(all_fold_rows).to_csv(exports / f"{stem}_folds.csv", index=False)
    (logs / f"{stem}.json").write_text(json.dumps(
        {"pooled": pooled_table.to_dict(orient="records"),
         "folds": all_fold_rows}, indent=2, default=float), encoding="utf-8")
    print("\nWalk-forward pooled comparison (2024-> rolling 2025 folds):")
    print(pooled_table.to_string(index=False,
                                 float_format=lambda v: f"{v:,.4f}"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--seeds", type=int, default=1, metavar="N",
        help="train every set N times with project.seed shifted by 0..N-1 and "
             "report mean +/- SE (default 1 = the published single-seed run)",
    )
    parser.add_argument(
        "--ablations", action="store_true",
        help="also train the leave-one-out microstructure ladder "
             "(micro_minus_<feature> for each extra column)",
    )
    parser.add_argument(
        "--early-stopping", action="store_true",
        help="enable chronological early stopping (last 20%% of the training "
             "panel held out, patience 25 epochs, best state restored + "
             "full-sample consequent refit). Off by default so the canonical "
             "numbers are unchanged.",
    )
    parser.add_argument(
        "--walk-forward", type=int, default=0, metavar="N_FOLDS",
        help="instead of the single 2024->2025 split, evaluate every set with "
             "N_FOLDS rolling retraining folds over the test year (each fold "
             "refits on all rows strictly before its block); reports per-fold "
             "metrics plus a pooled table with DSR on the winner's spread leg",
    )
    args = parser.parse_args()

    cfg = load_config()
    epochs = int(args.epochs or cfg["tsk"]["epochs"])
    n_seeds = max(1, int(args.seeds))
    base_seed = int(cfg.get("project", {}).get("seed", 0))
    sets = dict(DEFAULT_SETS)
    if args.ablations:
        sets.update(ABLATION_SETS)
    if args.early_stopping:
        sets = {name: {**spec, "early_stopping": True} for name, spec in sets.items()}

    if args.walk_forward > 0:
        _run_walk_forward(cfg, sets, epochs, int(args.walk_forward))
        return

    # per-row records: one per (set, seed); final table aggregates over seeds.
    rows, scored = [], {}
    for name, spec in sets.items():
        per_seed = []
        for k in range(n_seeds):
            seed = base_seed + k
            variant_cfg = _seeded_cfg(cfg, seed) if n_seeds > 1 else cfg
            model, panel, row = fit_variant(variant_cfg, spec, epochs)
            row.update({"set": name, "seed": seed})
            rows.append(row)
            per_seed.append(row)
            scored[(name, seed)] = panel
            print(f"{name:20} seed={seed}  inputs={row['n_inputs']}  "
                  f"IC={row['pooled_rank_ic']:+.4f}  RMSE={row['rmse']:.3f}  "
                  f"Sharpe15={row['net_sharpe_15bps']:+.3f}")
        if n_seeds > 1:
            ics = [r["pooled_rank_ic"] for r in per_seed]
            sharpes = [r["net_sharpe_15bps"] for r in per_seed]
            print(f"{name:20} IC spread over {n_seeds} seeds: "
                  f"[{min(ics):+.3f}, {max(ics):+.3f}]  "
                  f"Sharpe spread: [{min(sharpes):+.3f}, {max(sharpes):+.3f}]")

    frame = pd.DataFrame(rows)
    if n_seeds == 1:
        table = frame[
            ["set", "n_inputs", "n_free_params", "epochs_run", "rmse", "mae",
             "pooled_rank_ic", "monthly_mark_ic", "quarterly_mark_ic",
             "net_sharpe_15bps", "spread_sharpe_15bps"]
        ]
    else:
        parts = []
        for name, group in frame.groupby("set", sort=False):
            record = {
                "set": name,
                "n_inputs": int(group["n_inputs"].iloc[0]),
                "n_free_params": int(group["n_free_params"].iloc[0]),
                "n_seeds": int(group["seed"].nunique()),
            }
            for m in METRIC_COLUMNS:
                values = group[m].to_numpy(dtype=float)
                record[f"{m}_mean"] = float(np.nanmean(values))
                record[f"{m}_se"] = float(np.nanstd(values, ddof=1) / np.sqrt(len(values))) \
                    if len(values) > 1 else float("nan")
            parts.append(record)
        table = pd.DataFrame(parts)

    # ---- Deflated Sharpe ratio (Bailey & Lopez de Prado, 2014) --------------
    # The winning set was selected after searching this grid of feature sets x
    # seeds; DSR discounts the observed spread-portfolio Sharpe by the expected
    # maximum over `n_trials` independent trials, so the headline number is
    # corrected for selection. Evaluated on the canonical seed for single-seed
    # runs and on the first seed otherwise (per-seed DSRs go to the runs CSV).
    winner = max(sets, key=lambda s: frame.loc[frame["set"] == s,
                                               "pooled_rank_ic"].mean())
    win_rows = frame[frame["set"] == winner]
    first_seed = int(win_rows["seed"].iloc[0])
    win_panel = scored.get((winner, first_seed))
    dsr_record = {}
    if win_panel is not None:
        n_trials = len(sets) * n_seeds
        dsr_record = deflated_sharpe_ratio(
            _net_spread_excess(win_panel, cfg), n_trials=n_trials, periodicity=12)
        dsr_record.update({"winner": winner, "n_trials": n_trials})
        print(f"\nDeflated Sharpe ({winner}, n_trials={n_trials}): "
              f"DSR={dsr_record['dsr']:.3f}  SR_obs={dsr_record['sharpe_obs']:+.3f}  "
              f"SR*={dsr_record['sr_star']:+.3f}")

    exports = Path(cfg["paths"]["csv_exports"])
    logs = Path(cfg["paths"]["logs"])
    exports.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    stem = "feature_set_comparison" if n_seeds == 1 else f"feature_set_comparison_{n_seeds}seed"
    table.to_csv(exports / f"{stem}.csv", index=False)
    log_payload = {"table": table.to_dict(orient="records"),
                   "deflated_sharpe": dsr_record}
    (logs / f"{stem}.json").write_text(
        json.dumps(log_payload, indent=2), encoding="utf-8"
    )
    if n_seeds == 1:  # legacy artefacts stay exactly where they were
        scored[("full", base_seed)].to_csv(exports / "full_featureset_scores.csv",
                                           index=False)
    else:
        frame["dsr"] = [
            deflated_sharpe_ratio(
                _net_spread_excess(scored[(r["set"], r["seed"])], cfg),
                n_trials=len(sets) * n_seeds)["dsr"]
            for _, r in frame.iterrows()
        ]
        frame.to_csv(exports / "feature_set_comparison_runs.csv", index=False)

    print(f"\nFeature-set comparison (2025 out-of-sample, {n_seeds} seed(s)):")
    print(table.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))


if __name__ == "__main__":
    main()