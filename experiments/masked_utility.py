"""Utility cost of masking on *deployed* outputs (Referee Report, Analysis 4).

The ablation reports identical RMSE/IC/Sharpe across masking variants because
the scoring metrics are computed on unmasked scores. This script recomputes
every headline metric on the **masked** deployed path (Pre-AGM rows masked,
everything else untouched), for both the fixed alpha=0.05 scheme and the
adaptive epsilon=2.0 scheme, over independent masking draws:

* pooled + monthly Rank IC (score vs R30) on masked outputs;
* net Sharpe of the monthly top-3 portfolio on masked outputs;
* which monthly rebalances actually fall inside armed windows;
* noise-to-signal ratio of the masked rows (referee's ~0.9 reading).

Example
-------
python experiments/masked_utility.py --draws 50
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.backtest import monthly_rank_ic, net_sharpe, pooled_rank_ic, top_n_portfolio_returns
from src.data_loader import FEATURE_COLUMNS, build_panel, load_config
from src.masking import adaptive_masked_firing_scores, masked_scores
from src.TSK_engine import TSKFuzzySystem



def rebalance_flags(panel: pd.DataFrame) -> list[dict]:
    """First trading day of each month: how many names sit in an armed window?"""
    out = []
    for _stamp, group in panel.groupby(pd.Grouper(key="date", freq="MS")):
        first_day = group["date"].min()
        day_rows = panel[panel["date"] == first_day]
        out.append({"date": str(first_day.date()),
                    "n_masked_names": int((day_rows["pre_agm"] > 0).sum()),
                    "n_names": int(day_rows["ticker"].nunique())})
    return out



def run(cfg: dict, model: TSKFuzzySystem, draws: int, seed: int,
        epsilon: float, alpha_max: float) -> dict:
    panel = build_panel(cfg, split="test")
    X = panel[list(FEATURE_COLUMNS)].to_numpy(dtype=float)
    base = model.score(X)
    flags = panel["pre_agm"].to_numpy(dtype=float)
    in_window = flags > 0.0
    labelled = panel["Y_target"].notna().to_numpy()
    top_n = int(cfg["backtest"]["top_n"])
    bps = int(cfg["backtest"]["cost_tiers_bps"][0])
    y_target = panel["Y_target"].to_numpy(dtype=float)
    valid = labelled & np.isfinite(y_target)

    report: dict = {
        "n_masked_rows": int(in_window.sum()),
        "rebalances": rebalance_flags(panel),
        "base_signal_sd_in_window": float(base[in_window].std()),
        "schemes": {},
    }
    for scheme in ("fixed", "adaptive"):
        rows = []
        for d in range(draws):
            rng = np.random.default_rng(seed + d)
            if scheme == "fixed":
                masked = masked_scores(model, X, flags,
                                       alpha=float(cfg["masking"]["alpha"]), rng=rng)
            else:
                masked, _, _ = adaptive_masked_firing_scores(
                    model, X, epsilon=epsilon, alpha_max=alpha_max, rng=rng)
                masked = np.where(in_window, masked, base)
            delta = masked - base
            scored = panel.assign(Y_attr=masked)
            subset_masked = masked[valid]
            port = top_n_portfolio_returns(scored, top_n=top_n)
            rows.append({
                "pooled_ic": pooled_rank_ic(scored),
                "monthly_ic": monthly_rank_ic(scored)[0],
                "rmse": float(np.sqrt(np.mean(
                    (subset_masked - y_target[valid]) ** 2))),
                "net_sharpe": net_sharpe(port, cost_bps=bps),
                "noise_sd": float(delta[in_window].std()),
                "signed_bias": float(delta[in_window].mean()),
                "mean_abs_dev": float(np.abs(delta[in_window]).mean()),
            })
        df = pd.DataFrame(rows)
        sd_noise = float(df["noise_sd"].mean())
        report["schemes"][scheme] = {
            "pooled_ic_mean": float(df["pooled_ic"].mean()),
            "pooled_ic_sd": float(df["pooled_ic"].std(ddof=1)),
            "monthly_ic_mean": float(df["monthly_ic"].mean()),
            "monthly_ic_sd": float(df["monthly_ic"].std(ddof=1)),
            "rmse_mean": float(df["rmse"].mean()),
            "net_sharpe_mean": float(df["net_sharpe"].mean()),
            "net_sharpe_sd": float(df["net_sharpe"].std(ddof=1)),
            "noise_sd": sd_noise,
            "signed_bias": float(df["signed_bias"].mean()),
            "mean_abs_dev": float(df["mean_abs_dev"].mean()),
            "noise_to_signal": sd_noise / max(float(base[in_window].std()), 1e-12),
        }
    scored0 = panel.assign(Y_attr=base)
    port0 = top_n_portfolio_returns(scored0, top_n=top_n)
    report["unmasked"] = {
        "pooled_ic": pooled_rank_ic(scored0),
        "monthly_ic": monthly_rank_ic(scored0)[0],
        "rmse": float(np.sqrt(np.mean((base[valid] - y_target[valid]) ** 2))),
        "net_sharpe": net_sharpe(port0, cost_bps=bps),
    }
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="masked-output utility cost (analysis 4)")
    ap.add_argument("--draws", type=int, default=50)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--epsilon", type=float, default=2.0)
    ap.add_argument("--alpha-max", type=float, default=1.0)
    args = ap.parse_args()

    cfg = load_config()
    model = TSKFuzzySystem.load(Path(cfg["paths"]["models"]) / "panel_anfis.json")
    report = run(cfg, model, args.draws, args.seed, args.epsilon, args.alpha_max)

    logs = Path(cfg["paths"]["logs"])
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "masked_utility.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")

    print(f"masked rows: {report['n_masked_rows']}  "
          f"signal SD in window: {report['base_signal_sd_in_window']:.3f}")
    print("monthly rebalances vs armed windows:")
    for r in report["rebalances"]:
        mark = "  <-- MASKED" if r["n_masked_names"] else ""
        print(f"  {r['date']}: {r['n_masked_names']}/{r['n_names']} names{mark}")
    print(f"\nunmasked: pooled_ic={report['unmasked']['pooled_ic']:.3f} "
          f"monthly_ic={report['unmasked']['monthly_ic']:.3f} "
          f"sharpe={report['unmasked']['net_sharpe']:.3f}")
    for scheme, r in report["schemes"].items():
        print(f"{scheme:9s}: pooled_ic={r['pooled_ic_mean']:.3f}"
              f"+/-{r['pooled_ic_sd']:.3f} "
              f"monthly_ic={r['monthly_ic_mean']:.3f}"
              f"+/-{r['monthly_ic_sd']:.3f} "
              f"sharpe={r['net_sharpe_mean']:.3f}"
              f"+/-{r['net_sharpe_sd']:.3f} "
              f"noise_sd={r['noise_sd']:.2f} "
              f"n/s={r['noise_to_signal']:.2f} "
              f"bias={r['signed_bias']:+.3f}")
    print(f"\nfull report: {logs / 'masked_utility.json'}")


if __name__ == "__main__":
    main()

def rebalance_flags(panel: pd.DataFrame) -> list[dict]:
    """First trading day of each month: how many names sit in an armed window?"""
    out = []
    for stamp, group in panel.groupby(pd.Grouper(key="date", freq="MS")):
        first_day = group["date"].min()
        day_rows = panel[panel["date"] == first_day]
        out.append({"date": str(first_day.date()),
                    "n_masked_names": int((day_rows["pre_agm"] > 0).sum()),
                    "n_names": int(day_rows["ticker"].nunique())})
    return out
