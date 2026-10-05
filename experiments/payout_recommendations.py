"""Payout recommendations (Section 4.2, Table 1 of the paper).

Maps the mean out-of-sample attractiveness score of each equity through
``PR = psi(Y_attr) * Phi(Net Debt / EBITDA, FCFE)`` using the latest reported
fundamentals, and compares against the actual board decisions recorded in
``data/raw/financials/actual_decisions.csv`` when that file exists.

Example
-------
python experiments/payout_recommendations.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.corporate_payout import payout_table
from src.data_loader import FEATURE_COLUMNS, build_panel, load_config, load_financials
from src.TSK_engine import TSKFuzzySystem


def main() -> None:
    cfg = load_config()
    model_path = Path(cfg["paths"]["models"]) / "panel_anfis.json"
    if not model_path.exists():
        raise SystemExit(f"model not found at {model_path} -- run train_anfis_panel.py first")
    model = TSKFuzzySystem.load(model_path)

    test = build_panel(cfg, split="test")
    scored = test.assign(Y_attr=model.score(test[list(FEATURE_COLUMNS)].to_numpy(dtype=float)))

    scores = scored.groupby("ticker")["Y_attr"].mean().to_frame()
    financials = load_financials(cfg)
    table = payout_table(
        financials, scores,
        kappa=float(cfg["payout"]["kappa"]), tau=float(cfg["payout"]["tau"]),
    )
    table["PR_target_pct"] = (table["PR_target"] * 100.0).round(1)

    exports = Path(cfg["paths"]["csv_exports"])
    exports.mkdir(parents=True, exist_ok=True)
    table.to_csv(exports / "payout_recommendations.csv", index=False)
    (Path(cfg["paths"]["logs"]) / "payout_recommendations.json").write_text(
        json.dumps(table.to_dict(orient="records"), indent=2, default=str), encoding="utf-8"
    )

    latest = financials.groupby("ticker").tail(1)[
        ["ticker", "period", "period_year", "ebitda", "net_debt", "fcfe", "is_annual"]
    ]
    print("Latest reported fundamentals:")
    print(latest.to_string(index=False))
    print("\nPayout recommendations:")
    print(table.to_string(index=False, float_format=lambda v: f"{v:,.1f}"))


if __name__ == "__main__":
    main()
