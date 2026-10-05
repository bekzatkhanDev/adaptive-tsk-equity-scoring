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
from src.backtest import mark_rank_ic, net_sharpe, pooled_rank_ic, top_n_portfolio_returns
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
                  "monthly_mark_ic", "quarterly_mark_ic")


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
    trainer.fit(X, y, epochs=int(epochs), verbose_every=0)

    scored = test.assign(Y_attr=model.score(test[columns].to_numpy(dtype=float)))
    labelled = scored.dropna(subset=["Y_target"])
    prediction = labelled["Y_attr"].to_numpy(dtype=float)
    target = labelled["Y_target"].to_numpy(dtype=float)
    portfolio = top_n_portfolio_returns(scored, top_n=int(cfg["backtest"]["top_n"]))
    monthly_ic = mark_rank_ic(scored, freq="monthly", horizon=21)[0]
    quarterly_ic = mark_rank_ic(scored, freq="quarterly", horizon=63)[0]
    return model, scored, {
        "n_inputs": len(columns),
        "features": ",".join(columns),
        "n_free_params": int(model.n_free_params),
        "rmse": float(np.sqrt(np.mean((prediction - target) ** 2))),
        "mae": float(np.mean(np.abs(prediction - target))),
        "pooled_rank_ic": pooled_rank_ic(scored),
        "net_sharpe_15bps": net_sharpe(portfolio, cost_bps=cfg["backtest"]["cost_tiers_bps"][0]),
        "monthly_mark_ic": monthly_ic,
        "quarterly_mark_ic": quarterly_ic,
    }


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
    args = parser.parse_args()

    cfg = load_config()
    epochs = int(args.epochs or cfg["tsk"]["epochs"])
    n_seeds = max(1, int(args.seeds))
    base_seed = int(cfg.get("project", {}).get("seed", 0))
    sets = dict(DEFAULT_SETS)
    if args.ablations:
        sets.update(ABLATION_SETS)

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
            ["set", "n_inputs", "n_free_params", "rmse", "mae", "pooled_rank_ic",
             "monthly_mark_ic", "quarterly_mark_ic", "net_sharpe_15bps"]
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

    exports = Path(cfg["paths"]["csv_exports"])
    logs = Path(cfg["paths"]["logs"])
    exports.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    stem = "feature_set_comparison" if n_seeds == 1 else f"feature_set_comparison_{n_seeds}seed"
    table.to_csv(exports / f"{stem}.csv", index=False)
    (logs / f"{stem}.json").write_text(
        json.dumps(table.to_dict(orient="records"), indent=2), encoding="utf-8"
    )
    if n_seeds == 1:  # legacy artefacts stay exactly where they were
        scored[("full", base_seed)].to_csv(exports / "full_featureset_scores.csv",
                                           index=False)
    else:
        frame.to_csv(exports / "feature_set_comparison_runs.csv", index=False)

    print(f"\nFeature-set comparison (2025 out-of-sample, {n_seeds} seed(s)):")
    print(table.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))


if __name__ == "__main__":
    main()