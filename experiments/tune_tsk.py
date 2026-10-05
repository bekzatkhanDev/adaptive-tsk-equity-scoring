"""Chronological hyper-parameter scan for the TSK/ANFIS scorer (Tier-1).

Fits each candidate on 2024-H1 and validates on 2024-H2 -- the internal split the
config comment commits to -- maximising validated Rank IC subject to
Proposition 1's premise (empirical ``C = max|y_k - Y_raw| <= 100``). The grid
spans ``n_rules`` x ``cluster_radius`` x ``ridge_lambda``; the winning feasible
row is printed as a ready-to-paste ``config.yaml`` fragment.

Example
-------
python experiments/tune_tsk.py --epochs 150
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy import stats

from src.anfis_trainer import ANFISTrainer
from src.data_loader import FEATURE_COLUMNS, build_panel, load_config
from src.TSK_engine import TSKFuzzySystem

DEFAULT_GRID = {
    "n_rules": [3, 5, 7],
    "cluster_radius": [0.3, 0.4, 0.5, 0.6],
    "ridge_lambda": [1.0, 10.0, 100.0],
}


def empirical_C(model: TSKFuzzySystem, X: np.ndarray) -> float:
    """Proposition-1 constant on the deployed path (``C = max|y_k - Y_raw|``)."""
    weights = model._normalize(model._log_firing(X))
    rules = model.rule_outputs(X)
    if model.clip_rule_outputs:
        rules = np.clip(rules, 0.0, model.score_scale)
    aggregate = np.sum(weights * rules, axis=1)
    return float(np.abs(rules - aggregate[:, None]).max())


def fit_eval(cfg, h1, h2, n_rules, radius, ridge, epochs) -> dict:
    """Train one candidate on H1, score it on H2."""
    x1 = h1[list(FEATURE_COLUMNS)].to_numpy(dtype=float)
    y1 = h1["Y_target"].to_numpy(dtype=float)
    x2 = h2[list(FEATURE_COLUMNS)].to_numpy(dtype=float)
    y2 = h2["Y_target"].to_numpy(dtype=float)

    tsk_cfg = cfg["tsk"]
    model = TSKFuzzySystem(
        n_inputs=len(FEATURE_COLUMNS),
        n_rules=int(n_rules),
        sigma_floor=float(cfg["target"]["sigma_floor"]),
        firing_floor=float(cfg["target"]["firing_floor"]),
        neutral_score=float(cfg["target"]["neutral_score"]),
        score_scale=float(cfg["target"]["score_scale"]),
        clip_rule_outputs=bool(tsk_cfg.get("clip_rule_outputs", True)),
        feature_names=list(FEATURE_COLUMNS),
    )
    model.init_from_data(x1, radius=float(radius), ratio=float(tsk_cfg["cluster_ratio"]))
    trainer = ANFISTrainer(model, lr=float(tsk_cfg["premise_lr"]), ridge=float(ridge),
                           grad_clip=float(tsk_cfg["grad_clip"]))
    trainer.fit(x1, y1, epochs=int(epochs), verbose_every=0)

    prediction = model.predict(x2)
    ic = float(stats.spearmanr(prediction, y2).statistic) if np.std(prediction) > 0 else float("nan")
    return {
        "n_rules": int(n_rules),
        "cluster_radius": float(radius),
        "ridge_lambda": float(ridge),
        "val_rank_ic": ic,
        "val_rmse": float(np.sqrt(np.mean((prediction - y2) ** 2))),
        "val_mae": float(np.mean(np.abs(prediction - y2))),
        "empirical_C": empirical_C(model, x2),
        "final_loss": trainer.history.train_loss[-1] if trainer.history.train_loss else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=None, help="default: config tsk.epochs")
    parser.add_argument("--grid", default=None, help="path to a JSON grid (optional)")
    args = parser.parse_args()

    cfg = load_config()
    epochs = int(args.epochs or cfg["tsk"]["epochs"])
    grid = json.loads(Path(args.grid).read_text()) if args.grid else DEFAULT_GRID

    panel = build_panel(cfg, split="train").dropna(subset=["Y_target"])
    train_lo = pd.Timestamp(cfg["experiment"]["train_start"])
    train_hi = pd.Timestamp(cfg["experiment"]["train_end"])
    midpoint = train_lo + (train_hi - train_lo) / 2
    h1 = panel[panel["date"] <= midpoint]
    h2 = panel[panel["date"] > midpoint]
    if h1.empty or h2.empty:
        raise SystemExit("need early/late training halves -- check the train window and data")
    print(f"validation split: fit {train_lo.date()}..{midpoint.date()} | "
          f"validate {midpoint.date() + pd.Timedelta(days=1)}..{train_hi.date()}")

    rows = []
    for n_rules, radius, ridge in product(
        grid["n_rules"], grid["cluster_radius"], grid["ridge_lambda"]
    ):
        row = fit_eval(cfg, h1, h2, n_rules, radius, ridge, epochs)
        rows.append(row)
        print(f"K={n_rules} r={radius} ridge={ridge:<6}  "
              f"val_IC={row['val_rank_ic']:+.3f}  C={row['empirical_C']:.1f}")

    table = pd.DataFrame(rows).sort_values("val_rank_ic", ascending=False).reset_index(drop=True)
    logs = Path(cfg["paths"]["logs"])
    logs.mkdir(parents=True, exist_ok=True)
    table.to_csv(logs / "tsk_tuning.csv", index=False)

    feasible = table[table["empirical_C"] <= 100.0]
    best = (feasible if len(feasible) else table).iloc[0]
    print("\nTop candidates:")
    print(table.head(10).to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
    print("\nBest feasible config (C <= 100):")
    print(json.dumps({
        "n_rules": int(best["n_rules"]),
        "cluster_radius": float(best["cluster_radius"]),
        "ridge_lambda": float(best["ridge_lambda"]),
    }, indent=2))


if __name__ == "__main__":
    main()
