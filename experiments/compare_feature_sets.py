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

Example
-------
python experiments/compare_feature_sets.py
"""

from __future__ import annotations

import argparse
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

SETS = {
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


def fit_variant(cfg: dict, spec: dict, epochs: int):
    """Train one feature-set variant and score the 2025 out-of-sample panel."""
    variant = json.loads(json.dumps(cfg))  # deep copy
    variant["features"].update({
        "extra_features": spec["extra_features"],
        "fundamentals": spec["fundamentals"],
    })
    variant["tsk"]["input_scaling"] = spec["input_scaling"]

    columns = list(feature_columns(variant))
    train = build_panel(variant, split="train").dropna(subset=["Y_target"])
    test = build_panel(variant, split="test")

    X = train[columns].to_numpy(dtype=float)
    y = train["Y_target"].to_numpy(dtype=float)
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
    args = parser.parse_args()

    cfg = load_config()
    epochs = int(args.epochs or cfg["tsk"]["epochs"])

    rows, scored = [], {}
    for name, spec in SETS.items():
        model, panel, row = fit_variant(cfg, spec, epochs)
        row["set"] = name
        rows.append(row)
        scored[name] = panel
        print(f"{name:14} inputs={row['n_inputs']}  IC={row['pooled_rank_ic']:+.4f}  "
              f"RMSE={row['rmse']:.3f}  Sharpe15={row['net_sharpe_15bps']:+.3f}")

    table = pd.DataFrame(rows)[
        ["set", "n_inputs", "n_free_params", "rmse", "mae", "pooled_rank_ic",
         "monthly_mark_ic", "quarterly_mark_ic", "net_sharpe_15bps"]
    ]
    exports = Path(cfg["paths"]["csv_exports"])
    logs = Path(cfg["paths"]["logs"])
    exports.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    table.to_csv(exports / "feature_set_comparison.csv", index=False)
    (logs / "feature_set_comparison.json").write_text(
        json.dumps(table.to_dict(orient="records"), indent=2), encoding="utf-8"
    )
    scored["full"].to_csv(exports / "full_featureset_scores.csv", index=False)

    print("\nFeature-set comparison (2025 out-of-sample):")
    print(table.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))


if __name__ == "__main__":
    main()