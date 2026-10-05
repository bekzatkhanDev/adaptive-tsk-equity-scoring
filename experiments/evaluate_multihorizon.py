"""Multi-horizon and period-mark evaluation (Tier-1 of the improvement plan).

Reuses the Section-4 protocol (train 2024, test 2025) to report:

* per-horizon fit (RMSE / MAE against ``Y_target_{h}``) and pooled / per-asset
  Rank IC of ``Y_attr`` against the raw forward return ``R{h}``;
* the same statistics for a four-parameter OLS baseline on identical features;
* cross-sectional **period-mark** Rank IC -- the weekly / monthly / quarterly /
  annual "mark" a corporate board would see, computed as the aggregate of the
  daily scores over a period -- predicting the *next* period's forward return.

Example
-------
python experiments/evaluate_multihorizon.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.backtest import mark_rank_ic, per_asset_rank_ic, pooled_rank_ic
from src.data_loader import FEATURE_COLUMNS, build_panel, load_config, target_horizons
from src.TSK_engine import TSKFuzzySystem

# (period alias, matching trading-day horizon) for the mark statistics.
MARK_PERIODS = [("weekly", 5), ("monthly", 21), ("quarterly", 63), ("annual", 252)]


def ols_predict(train: pd.DataFrame, test: pd.DataFrame, y_col: str) -> np.ndarray:
    """Closed-form OLS on the shared features (fit on the training window)."""
    X = train[list(FEATURE_COLUMNS)].to_numpy(dtype=float)
    y = train[y_col].to_numpy(dtype=float)
    keep = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[keep], y[keep]
    design = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    return beta[0] + test[list(FEATURE_COLUMNS)].to_numpy(dtype=float) @ beta[1:]


def horizon_metrics(scored: pd.DataFrame, train: pd.DataFrame, horizon: int) -> dict:
    """Fit + ranking metrics for one horizon (TSK and OLS on the same labels).

    Horizons longer than the test window have no observable label; those rows are
    reported with ``n_labelled = 0`` and NaN fit metrics rather than dropped, so
    the report is explicit about what the data can and cannot support.
    """
    y_col, ret_col = f"Y_target_{horizon}", f"R{horizon}"
    labelled = scored.dropna(subset=[y_col])

    ols_scored = scored.assign(Y_attr_ols=ols_predict(train, scored, y_col))
    row = {
        "horizon_days": int(horizon),
        "n_labelled": int(len(labelled)),
        "rmse": float("nan"),
        "mae": float("nan"),
        "pooled_rank_ic": pooled_rank_ic(scored, ret_col=ret_col),
        "per_asset_mean_rank_ic": float("nan"),
        "ols_pooled_rank_ic": pooled_rank_ic(ols_scored, score_col="Y_attr_ols", ret_col=ret_col),
    }
    if len(labelled):
        prediction = labelled["Y_attr"].to_numpy(dtype=float)
        target = labelled[y_col].to_numpy(dtype=float)
        row["rmse"] = float(np.sqrt(np.mean((prediction - target) ** 2)))
        row["mae"] = float(np.mean(np.abs(prediction - target)))
        per_asset = per_asset_rank_ic(labelled, ret_col=ret_col).to_numpy(dtype=float)
        finite = per_asset[np.isfinite(per_asset)]
        if finite.size:
            row["per_asset_mean_rank_ic"] = float(finite.mean())
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None, help="panel model JSON")
    args = parser.parse_args()

    cfg = load_config()
    model_path = Path(args.model or Path(cfg["paths"]["models"]) / "panel_anfis.json")
    if not model_path.exists():
        raise SystemExit(f"model not found at {model_path} -- run train_anfis_panel.py first")
    model = TSKFuzzySystem.load(model_path)

    train = build_panel(cfg, split="train").dropna(subset=["Y_target"])
    test = build_panel(cfg, split="test")
    scored = test.assign(Y_attr=model.score(test[list(FEATURE_COLUMNS)].to_numpy(dtype=float)))

    horizons = [h for h in target_horizons(cfg) if f"Y_target_{h}" in scored.columns]
    table = pd.DataFrame(horizon_metrics(scored, train, h) for h in horizons)

    agg = str(cfg.get("target", {}).get("aggregation", "mean"))
    mark_rows = []
    for name, horizon in MARK_PERIODS:
        if f"R{horizon}" not in scored.columns:
            continue
        mean_ic, se, series = mark_rank_ic(scored, freq=name, horizon=horizon, agg=agg)
        mark_rows.append({
            "mark": name,
            "horizon_days": horizon,
            "aggregation": agg,
            "n_periods": int(len(series)),
            "mean_rank_ic": mean_ic,
            "se": se,
        })
    marks = pd.DataFrame(mark_rows)

    exports = Path(cfg["paths"]["csv_exports"])
    logs = Path(cfg["paths"]["logs"])
    exports.mkdir(parents=True, exist_ok=True)
    table.to_csv(exports / "multihorizon_metrics.csv", index=False)
    marks.to_csv(exports / "period_mark_metrics.csv", index=False)
    (logs / "multihorizon_report.json").write_text(
        json.dumps({"horizons": table.to_dict(orient="records"),
                    "marks": marks.to_dict(orient="records")}, indent=2, default=str),
        encoding="utf-8",
    )

    print("Per-horizon metrics (TSK vs OLS on identical labels):")
    print(table.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
    print("\nPeriod-mark Rank IC (aggregated score predicts the next period):")
    print(marks.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))


if __name__ == "__main__":
    main()
