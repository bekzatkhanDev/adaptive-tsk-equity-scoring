"""Pooled-panel TSK/ANFIS training across the seven KASE equities.

The unified rule base is fit on the pooled training window and evaluated
per-ticker on the test window. Model + metrics are written to ``outputs``.

Example
-------
python experiments/train_anfis_panel.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.anfis_trainer import ANFISTrainer
from src.data_loader import build_panel, feature_columns, load_config
from src.TSK_engine import TSKFuzzySystem

# Config-aware active input vector (equals the canonical three inputs unless the
# Tier-2 switches in ``config.yaml`` are turned on).
FEATURE_COLUMNS = feature_columns(load_config())


def train_panel(cfg: dict, epochs: int | None = None, verbose: bool = True):
    train = build_panel(cfg, split="train").dropna(subset=["Y_target"])
    test = build_panel(cfg, split="test")

    X = train[list(FEATURE_COLUMNS)].to_numpy(dtype=float)
    y = train["Y_target"].to_numpy(dtype=float)

    tsk_cfg = cfg["tsk"]
    model = TSKFuzzySystem(
        n_inputs=len(FEATURE_COLUMNS),
        n_rules=int(tsk_cfg["n_rules"]),
        sigma_floor=float(cfg["target"]["sigma_floor"]),
        firing_floor=float(cfg["target"]["firing_floor"]),
        neutral_score=float(cfg["target"]["neutral_score"]),
        score_scale=float(cfg["target"]["score_scale"]),
        clip_rule_outputs=bool(tsk_cfg.get("clip_rule_outputs", True)),
        input_scaling=str(tsk_cfg.get("input_scaling", "raw")),
        feature_names=list(FEATURE_COLUMNS),
    )
    model.init_from_data(X, radius=float(tsk_cfg["cluster_radius"]),
                         ratio=float(tsk_cfg["cluster_ratio"]))
    trainer = ANFISTrainer(
        model,
        lr=float(tsk_cfg["premise_lr"]),
        ridge=float(tsk_cfg["ridge_lambda"]),
        grad_clip=float(tsk_cfg["grad_clip"]),
    )
    trainer.fit(X, y, epochs=int(epochs or tsk_cfg["epochs"]),
                verbose_every=int(tsk_cfg["verbose_every"]) if verbose else 0)

    X_test = test[list(FEATURE_COLUMNS)].to_numpy(dtype=float)
    test = test.assign(Y_attr=model.score(X_test))
    metrics = {
        "n_train": int(len(X)),
        "n_rules": int(model.n_rules),
        "n_free_params": int(model.n_free_params),
        "final_loss": trainer.history.train_loss[-1] if trainer.history.train_loss else None,
        "per_ticker": {},
    }
    for ticker, rows in test.groupby("ticker"):
        finite = rows.dropna(subset=["Y_target"])
        if len(finite):
            metrics["per_ticker"][ticker] = trainer.evaluate(
                finite[list(FEATURE_COLUMNS)].to_numpy(dtype=float),
                finite["Y_target"].to_numpy(dtype=float),
            )
    return model, trainer, test, metrics


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    model, trainer, test, metrics = train_panel(cfg, args.epochs, not args.quiet)

    model.save(Path(cfg["paths"]["models"]) / "panel_anfis.json")
    trainer.history.to_frame().to_csv(
        Path(cfg["paths"]["logs"]) / "panel_training_history.csv", index=False
    )
    Path(cfg["paths"]["logs"]).joinpath("panel_train_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    test[["date", "ticker", "close", "R30", "Y_target", "Y_attr", "pre_agm"]].to_csv(
        Path(cfg["paths"]["csv_exports"]) / "panel_2025_scores.csv", index=False
    )
    if not args.quiet:
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
