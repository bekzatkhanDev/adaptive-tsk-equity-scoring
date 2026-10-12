"""Payout alignment with correct fiscal timing (Referee Report, Analysis 7).

Two corrections the referee specifies for Table ``tab:h1_validation``:

1. **Fiscal-year alignment.** The 2025 AGM decisions (May--June 2025) rule on
   *FY2024* payouts, and the information available at that date is the FY2024
   annual disclosure (plus quarters of 2024) -- not the FY2025 financials the
   paper's table uses. The model target is recomputed from FY2024 annual
   fundamentals, and the score is truncated to the pre-decision window
   (2025-01-01 .. AGM date) so no post-decision data leaks into the target.
2. **Fundamentals-only benchmark.** The score's marginal contribution is
   tested against a rule that ignores it entirely: ``PR = Phi(ND/EBITDA, FCFE)``
   (Eq. 13 alone). If the score adds nothing, the alignment table is a
   fundamentals screen with a decorative psi layer.

Actual decisions are the board ranges recorded in the paper's Table
(``data/raw/financials/actual_decisions.csv`` is absent in this repo); they
are treated as FY2024 payout decisions, which is when they were announced.

Example
-------
python experiments/payout_alignment.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.corporate_payout import phi, psi
from src.data_loader import (FEATURE_COLUMNS, build_panel, load_board_decisions,
                             load_config, load_financials)
from src.TSK_engine import TSKFuzzySystem

# Board decisions from the paper's Table (FY2024 payouts, announced at the
# 2025 AGMs). Ranges are kept as-is; the midpoint is used for gap metrics.
# Fallback only: when data/raw/financials/actual_decisions.csv exists it
# anchors the alignment instead (see _csv_actuals below).
ACTUAL = {
    "KMGZ": (0.50, 0.60),
    "KZAP": (0.75, 1.00),
    "KEGC": (0.65, 0.75),
    "KCEL": (0.00, 0.00),
    "AIRA": (0.00, 0.00),
    "KZTK": (0.00, 0.00),
    "KZTO": (0.85, 1.00),
}
AGM_2025 = {
    "KMGZ": "2025-05-16", "KZAP": "2025-05-22", "KEGC": "2025-05-14",
    "KZTO": "2025-05-15", "KCEL": "2025-06-13", "AIRA": "2025-06-20",
    "KZTK": "2025-05-29",
}


def score_truncated(model, cfg: dict) -> pd.DataFrame:
    """Mean score per ticker over 2025-01-01 .. AGM date (pre-decision only)."""
    panel = build_panel(cfg, split="test")
    panel = panel.assign(Y_attr=model.score(
        panel[list(FEATURE_COLUMNS)].to_numpy(dtype=float)))
    rows = []
    for ticker, cutoff in AGM_2025.items():
        sub = panel[(panel["ticker"] == ticker)
                    & (panel["date"] < pd.Timestamp(cutoff))]
        if len(sub):
            rows.append({"ticker": ticker,
                         "score_pre_agm_mean": float(sub["Y_attr"].mean()),
                         "score_full_year_mean": float(
                             panel.loc[panel["ticker"] == ticker,
                                       "Y_attr"].mean())})
    return pd.DataFrame(rows)


def _csv_actuals(cfg: dict) -> dict | None:
    """Approved FY2024 singles from actual_decisions.csv, or None.

    Returns ``{ticker: (lo, hi)}`` with ``lo == hi == approved`` whenever the
    CSV has a non-null approved value for every ticker of the ACTUAL fallback;
    otherwise None so the caller keeps the paper-transcribed ranges.
    """
    try:
        board = load_board_decisions(cfg, 2024).set_index("ticker")
    except Exception:
        return None
    if board.empty or not set(ACTUAL).issubset(set(board.index)):
        return None
    appr = board.loc[list(ACTUAL), "approved"]
    if bool(appr.isna().any()):
        return None
    return {t: (float(v), float(v)) for t, v in appr.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description="payout alignment (analysis 7)")
    ap.add_argument("--kappa", type=float, default=3.0)
    ap.add_argument("--tau", type=float, default=1.0)
    args = ap.parse_args()

    cfg = load_config()
    model = TSKFuzzySystem.load(Path(cfg["paths"]["models"]) / "panel_anfis.json")
    financials = load_financials(cfg)

    # FY2024 annual fundamentals: the disclosure available at the 2025 AGMs.
    fy24 = financials[financials["is_annual"] & (financials["period_year"] == 2024)]
    fy24 = fy24.set_index("ticker")

    scores = score_truncated(model, cfg).set_index("ticker")

    rows = []
    csv_actuals = _csv_actuals(cfg)
    actuals = csv_actuals if csv_actuals is not None else ACTUAL
    actuals_note = (
        "FY2024 approved singles from data/raw/financials/actual_decisions.csv"
        if csv_actuals is not None
        else "board ranges from the paper's Table; "
             "actual_decisions.csv absent in this repo"
    )
    for ticker, (lo, hi) in actuals.items():
        fund = fy24.loc[ticker]
        leverage = float(fund["net_debt"] / fund["ebitda"])
        fcfe = float(fund["fcfe"])
        y_pre = float(scores.loc[ticker, "score_pre_agm_mean"])
        y_full = float(scores.loc[ticker, "score_full_year_mean"])
        # Referee D: with psi = Y/100 and all scores near 50, psi is a ~0.5
        # constant scale; the fundamentals-only rule sets psi == 1.
        model_pre = float(psi(np.array([y_pre]))[0] * phi(
            np.array([leverage]), np.array([fcfe]), args.kappa, args.tau)[0])
        model_full = float(psi(np.array([y_full]))[0] * phi(
            np.array([leverage]), np.array([fcfe]), args.kappa, args.tau)[0])
        fund_only = float(phi(
            np.array([leverage]), np.array([fcfe]), args.kappa, args.tau)[0])
        mid = (lo + hi) / 2.0
        rows.append({
            "ticker": ticker,
            "fy2024_fcfe": fcfe,
            "fy2024_nd_ebitda": leverage,
            "score_pre_agm": y_pre,
            "score_full_year": y_full,
            "model_target_pre_agm": model_pre,
            "model_target_full_year": model_full,
            "fundamentals_only": fund_only,
            "actual_lo": lo, "actual_hi": hi, "actual_mid": mid,
            "abs_gap_model": abs(model_pre - mid),
            "abs_gap_fund_only": abs(fund_only - mid),
            "psi_pre_agm": y_pre / 100.0,
        })
    table = pd.DataFrame(rows).sort_values("ticker").reset_index(drop=True)

    def _align(row) -> str:
        if row["actual_lo"] == row["actual_hi"] == 0.0:
            return "trivial 0=0" if row["model_target_pre_agm"] < 0.01 else "above 0"
        if row["model_target_pre_agm"] < row["actual_lo"]:
            return "below range"
        if row["model_target_pre_agm"] > row["actual_hi"]:
            return "above range"
        return "in range"

    table["alignment"] = table.apply(_align, axis=1)
    summary = {
        "n_trivial_zero_alignments": int(((table["actual_mid"] == 0.0)
                                          & (table["alignment"] == "trivial 0=0")).sum()),
        "n_positive_payers": int((table["actual_mid"] > 0.0).sum()),
        "mean_abs_gap_model": float(table["abs_gap_model"].mean()),
        "mean_abs_gap_fundamentals_only": float(table["abs_gap_fund_only"].mean()),
        "psi_range": [float(table["psi_pre_agm"].min()),
                      float(table["psi_pre_agm"].max())],
        "score_range_pre_agm": [float(table["score_pre_agm"].min()),
                                float(table["score_pre_agm"].max())],
        "fy_basis": "FY2024 annual disclosure (available at the 2025 AGMs)",
        "score_basis": "mean score 2025-01-01 .. AGM date (pre-decision)",
        "actuals_note": actuals_note,
    }

    logs = Path(cfg["paths"]["logs"])
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "payout_alignment.json").write_text(
        json.dumps({"summary": summary, "table": table.to_dict(orient="records")},
                   indent=2, default=str), encoding="utf-8")

    print(table[["ticker", "fy2024_nd_ebitda", "score_pre_agm",
                 "model_target_pre_agm", "fundamentals_only",
                 "actual_lo", "actual_hi", "alignment"]]
          .to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
    print()
    for k, v in summary.items():
        print(f"{k}: {v}")
    print(f"\nfull report: {logs / 'payout_alignment.json'}")


if __name__ == "__main__":
    main()

# (stray top-level return removed; score_truncated returns at its natural end)
