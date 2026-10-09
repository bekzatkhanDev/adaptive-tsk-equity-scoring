"""Walk-forward validation across training/test year splits (review item 1).

Refits the canonical TSK/ANFIS protocol on expanding training windows and
scores each held-out year, so no single test year (2025) carries the whole
claim. Splits are chronological and strictly out-of-sample: with bars from
2022 on, the feasible folds are train 2022 -> test 2023, train 2022-2023 ->
test 2024, and the canonical train 2022-2024 -> test 2025. Each fold rebuilds
labels with training-window-only moments (no leakage) and runs the shared
ANFIS protocol (same K, radius, ridge, epochs as the deployed model).

Example
-------
python experiments/walk_forward.py --epochs 200
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
from src.backtest import (
    monthly_rank_ic,
    net_sharpe,
    pooled_rank_ic,
    t_ci95,
    top_n_portfolio_returns,
)
from src.data_loader import build_panel, feature_columns, load_config
from src.TSK_engine import TSKFuzzySystem

FOLDS = [
    ("2022-01-01", "2022-12-31", "2023-01-01", "2023-12-31"),
    ("2022-01-01", "2023-12-31", "2024-01-01", "2024-12-31"),
    ("2022-01-01", "2024-12-31", "2025-01-01", "2025-12-31"),
]


def _split_panel(cfg: dict, tr0: str, tr1: str, te0: str, te1: str):
    sub = copy.deepcopy(cfg)
    sub["experiment"]["train_start"] = tr0
    sub["experiment"]["train_end"] = tr1
    sub["experiment"]["test_start"] = te0
    sub["experiment"]["test_end"] = te1
    train = build_panel(sub, split="train").dropna(subset=["Y_target"])
    test = build_panel(sub, split="test")
    return train, test


def _fit(cfg: dict, feats: list[str], train: pd.DataFrame, epochs: int) -> TSKFuzzySystem:
    X = train[list(feats)].to_numpy(dtype=float)
    y = train["Y_target"].to_numpy(dtype=float)
    tsk_cfg = cfg["tsk"]
    model = TSKFuzzySystem(
        n_inputs=len(feats),
        n_rules=int(tsk_cfg["n_rules"]),
        sigma_floor=float(cfg["target"]["sigma_floor"]),
        firing_floor=float(cfg["target"]["firing_floor"]),
        neutral_score=float(cfg["target"]["neutral_score"]),
        score_scale=float(cfg["target"]["score_scale"]),
        clip_rule_outputs=bool(tsk_cfg.get("clip_rule_outputs", True)),
        input_scaling=str(tsk_cfg.get("input_scaling", "raw")),
        feature_names=list(feats),
    )
    model.init_from_data(X, radius=float(tsk_cfg["cluster_radius"]),
                         ratio=float(tsk_cfg["cluster_ratio"]))
    trainer = ANFISTrainer(model, lr=float(tsk_cfg["premise_lr"]),
                           ridge=float(tsk_cfg["ridge_lambda"]),
                           grad_clip=float(tsk_cfg["grad_clip"]))
    trainer.fit(X, y, epochs=int(epochs), verbose_every=0)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config()
    feats = feature_columns(cfg)
    epochs = int(args.epochs or cfg["tsk"]["epochs"])
    top_n = int(cfg["backtest"]["top_n"])

    rows = []
    for tr0, tr1, te0, te1 in FOLDS:
        train, test = _split_panel(cfg, tr0, tr1, te0, te1)
        if len(train) < 100 or len(test) < 50:
            rows.append({"train": f"{tr0}..{tr1}", "test_year": te0[:4],
                         "note": "insufficient bars", "n_train": len(train),
                         "n_test": len(test)})
            continue
        model = _fit(cfg, feats, train, epochs)
        scored = test.assign(Y_attr=model.score(test[list(feats)].to_numpy(dtype=float)))
        labelled = scored.dropna(subset=["Y_target"])
        port = top_n_portfolio_returns(scored, top_n=top_n)
        mean_m, se_m, _ = monthly_rank_ic(scored)
        rows.append({
            "train": f"{tr0}..{tr1}", "test_year": te0[:4],
            "n_train": int(len(train)), "n_test": int(len(test)),
            "n_labelled": int(len(labelled)),
            "rmse": float(np.sqrt(np.mean((labelled["Y_attr"] - labelled["Y_target"]) ** 2))) if len(labelled) else float("nan"),
            "pooled_rank_ic": pooled_rank_ic(scored),
            "monthly_rank_ic": mean_m, "monthly_rank_ic_se": se_m,
            "net_sharpe_15bps": net_sharpe(port, cost_bps=15.0),
        })
    table = pd.DataFrame(rows)
    logs = Path(cfg["paths"]["logs"])
    table.to_csv(logs / "walk_forward.csv", index=False)
    ics = table["monthly_rank_ic"].to_numpy(dtype=float)
    sharpes = table["net_sharpe_15bps"].to_numpy(dtype=float)
    summary = {
        "folds": rows,
        "monthly_ic_mean": float(np.nanmean(ics)) if len(ics) else float("nan"),
        "monthly_ic_t_ci95": [float(v) for v in t_ci95(ics)],
        "sharpe_mean": float(np.nanmean(sharpes)) if len(sharpes) else float("nan"),
        "sharpe_t_ci95": [float(v) for v in t_ci95(sharpes)],
    }
    (logs / "walk_forward.json").write_text(json.dumps(summary, indent=2, default=str))
    print(table.to_string(index=False))
    print(json.dumps({k: v for k, v in summary.items() if k != "folds"}, indent=2))


if __name__ == "__main__":
    main()
