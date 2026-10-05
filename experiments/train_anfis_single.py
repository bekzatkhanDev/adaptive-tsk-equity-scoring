"""Train the TSK/ANFIS scorer on a single ticker (Section 3.3.1 protocol).

Example
-------
python experiments/train_anfis_single.py --ticker KMGZ
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.anfis_trainer import ANFISTrainer
from src.data_loader import FEATURE_COLUMNS, build_panel, load_config
from src.TSK_engine import TSKFuzzySystem


def train_ticker(cfg: dict, ticker: str, epochs: int | None = None, verbose: bool = True):
    panel = build_panel(cfg, split="train").dropna(subset=["Y_target"])
    rows = panel[panel["ticker"] == ticker]
    if rows.empty:
        raise SystemExit(f"no training rows for {ticker} -- check the train window")

    X = rows[list(FEATURE_COLUMNS)].to_numpy(dtype=float)
    y = rows["Y_target"].to_numpy(dtype=float)

    tsk_cfg = cfg["tsk"]
    model = TSKFuzzySystem(
        n_inputs=len(FEATURE_COLUMNS),
        n_rules=int(tsk_cfg["n_rules"]),
        sigma_floor=float(cfg["target"]["sigma_floor"]),
        firing_floor=float(cfg["target"]["firing_floor"]),
        neutral_score=float(cfg["target"]["neutral_score"]),
        score_scale=float(cfg["target"]["score_scale"]),
        clip_rule_outputs=bool(tsk_cfg.get("clip_rule_outputs", True)),
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

    test = build_panel(cfg, split="test")
    test_rows = test[test["ticker"] == ticker]
    metrics = {
        "ticker": ticker,
        "n_train": int(len(X)),
        "n_rules": int(model.n_rules),
        "n_free_params": int(model.n_free_params),
    }
    if len(test_rows):
        metrics["test"] = trainer.evaluate(
            test_rows[list(FEATURE_COLUMNS)].to_numpy(dtype=float),
            test_rows["Y_target"].to_numpy(dtype=float),
        )
    return model, trainer, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    model, trainer, metrics = train_ticker(cfg, args.ticker, args.epochs, not args.quiet)

    models_dir = Path(cfg["paths"]["models"])
    logs_dir = Path(cfg["paths"]["logs"])
    model.save(models_dir / f"{args.ticker}_anfis.json")
    trainer.history.to_frame().to_csv(logs_dir / f"{args.ticker}_training_history.csv", index=False)
    (logs_dir / f"{args.ticker}_train_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    if not args.quiet:
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
