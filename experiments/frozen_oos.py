"""Frozen pre-2025 hyper-parameter OOS re-run (Referee Report, M3).

The deployed configuration (K=5, radius=0.5, ridge=10) was retained *after*
observing 2025 outcomes (config.yaml tsk comment), so the published OOS
numbers carry a test-informed selection. This script restores a pre-2025
conditional estimate:

1. Re-runs the chronological pre-2025 scan (fit early training half,
   validate late training half -- both strictly before 2025, 36 configs).
   The fit--validation boundary is not purged in this scan, so the
   selection is not fully leakage-free.
   and takes its feasible winner as the *frozen* spec.
2. Retrains that frozen spec on 2022--2024 with a 30-trading-day embargo
   at the train/test boundary (``purged_robustness.purge_train``).
3. Evaluates on 2025 with the identical metric stack as
   ``experiments/evaluate_2025_oos.py``, plus the extended 15/25/35/50/
   75/100/150-bps cost grid (M4).

The deployed spec under the same embargo trains alongside for an
apples-to-apples comparison, so the paper reports:

- frozen pre-2025 pick + test-boundary embargo = conditional OOS estimate
  (no 2025 look, but the fit--validation scan is not purged),
- deployed pick + embargo        = test-informed sensitivity row.

Example
-------
python experiments/frozen_oos.py
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

from experiments.evaluate_2025_oos import evaluate
from experiments.purged_robustness import fit_model, purge_train
from experiments.tune_tsk import DEFAULT_GRID, fit_eval
from src.backtest import net_sharpe
from src.data_loader import build_panel, load_config

# Documented frozen pick: best pre-2025 validation IC among feasible configs
# (empirical C <= 100). Selected without ever observing 2025.
FROZEN_SPEC = {"n_rules": 3, "cluster_radius": 0.4, "ridge_lambda": 100.0}
DEPLOYED_SPEC = {"n_rules": 5, "cluster_radius": 0.5, "ridge_lambda": 10.0}
EXTENDED_COST_TIERS = [15, 25, 35, 50, 75, 100, 150]


def frozen_scan(cfg: dict, epochs: int) -> pd.DataFrame:
    """Re-run the pre-2025 chronological scan; return the sorted table."""
    panel = build_panel(cfg, split="train").dropna(subset=["Y_target"])
    train_lo = pd.Timestamp(cfg["experiment"]["train_start"])
    train_hi = pd.Timestamp(cfg["experiment"]["train_end"])
    midpoint = train_lo + (train_hi - train_lo) / 2
    h1 = panel[panel["date"] <= midpoint]
    h2 = panel[panel["date"] > midpoint]
    rows = [
        fit_eval(cfg, h1, h2, n_rules, radius, ridge, epochs)
        for n_rules in DEFAULT_GRID["n_rules"]
        for radius in DEFAULT_GRID["cluster_radius"]
        for ridge in DEFAULT_GRID["ridge_lambda"]
    ]
    return pd.DataFrame(rows).sort_values(
        "val_rank_ic", ascending=False).reset_index(drop=True)


def spec_fit(cfg: dict, train: pd.DataFrame, spec: dict) -> "TSKFuzzySystem":
    """Fit a model with an explicit K/radius/ridge spec on a (purged) panel."""
    local = copy.deepcopy(cfg)
    local["tsk"]["n_rules"] = int(spec["n_rules"])
    local["tsk"]["cluster_radius"] = float(spec["cluster_radius"])
    local["tsk"]["ridge_lambda"] = float(spec["ridge_lambda"])
    return fit_model(local, train)


def main() -> None:
    ap = argparse.ArgumentParser(description="frozen pre-2025 spec + embargo (M3)")
    ap.add_argument("--horizon", type=int, default=None,
                    help="embargo in trading days (default: target.horizon=30)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--skip-scan", action="store_true",
                    help="reuse the frozen spec without re-running the scan")
    args = ap.parse_args()

    cfg = load_config()
    horizon = int(args.horizon or cfg["target"]["horizon"])
    epochs = int(args.epochs or cfg["tsk"]["epochs"])
    logs = Path(cfg["paths"]["logs"])
    report: dict = {
        "embargo_trading_days": horizon,
        "frozen_selection_rule": (
            "chronological pre-2025 validation IC (fit early training half, "
            "validate late training half; both strictly before 2025), "
            "without a fit-validation purge; feasible under empirical C <= 100; "
            "the 36-point scan never observes 2025"),
    }

    # 1. Frozen scan on pre-2025 data only.
    if not args.skip_scan:
        table = frozen_scan(cfg, epochs)
        feasible = table[table["empirical_C"] <= 100.0]
        best = (feasible if len(feasible) else table).iloc[0]
        report["frozen_scan_winner"] = {
            "n_rules": int(best["n_rules"]),
            "cluster_radius": float(best["cluster_radius"]),
            "ridge_lambda": float(best["ridge_lambda"]),
            "val_rank_ic": float(best["val_rank_ic"]),
            "feasible_rank": int(
                feasible.reset_index(drop=True).index[
                    (feasible["n_rules"] == best["n_rules"])
                    & (feasible["cluster_radius"] == best["cluster_radius"])
                    & (feasible["ridge_lambda"] == best["ridge_lambda"])
                ][0] + 1),
            "n_feasible": int(len(feasible)),
        }
        table.to_csv(logs / "frozen_tuning.csv", index=False)
        committed = {k: report["frozen_scan_winner"][k]
                     for k in ("n_rules", "cluster_radius", "ridge_lambda")}
    else:
        committed = dict(FROZEN_SPEC)
    report["frozen_spec"] = committed
    report["deployed_spec"] = dict(DEPLOYED_SPEC)

    # 2. Embargoed training panel: purge train rows whose forward horizon
    #    window crosses the train/test boundary (leakage control).
    train = build_panel(cfg, split="train")
    test = build_panel(cfg, split="test")
    full = pd.concat([train, test], ignore_index=True)
    purged, cutoff = purge_train(train, full, horizon)
    report.update({
        "n_train_original": int(len(train.dropna(subset=["Y_target"]))),
        "n_train_purged": int(len(purged.dropna(subset=["Y_target"]))),
        "purge_cutoff": cutoff,
    })

    # 3. Train + evaluate both specs under the same embargo.
    train_clean = train.dropna(subset=["Y_target"])
    for name, spec in (("frozen_purged", committed),
                       ("deployed_purged", DEPLOYED_SPEC)):
        model = spec_fit(cfg, purged, spec)
        result_report, scored, portfolio = evaluate(cfg, model, train_clean, test)
        port_rets = portfolio["portfolio_return"].to_numpy(dtype=float)
        tiers = {str(b): float(net_sharpe(portfolio, cost_bps=b))
                 for b in EXTENDED_COST_TIERS}
        membership = [set(p.split(",")) for p in portfolio["picks"]]
        changes = [
            len(membership[i] ^ membership[i - 1]) / max(len(membership[i]), 1)
            for i in range(1, len(membership))
        ]
        monthly_turnover = float(np.mean(changes)) / 2.0 if changes else 0.0
        monthly_cost_15 = monthly_turnover * 15.0 / 1e4
        net_port_rets_15 = port_rets - monthly_cost_15
        tsk = result_report["tsk"]
        report[name] = {
            "spec": spec,
            "monthly_rank_ic": tsk["monthly_rank_ic_mean"],
            "monthly_rank_ic_se": tsk["monthly_rank_ic_se"],
            "monthly_rank_ic_per_month":
                result_report["monthly_rank_ic_per_month"],
            "monthly_rank_ic_bootstrap_ci95":
                _date_clustered_bootstrap_ic(scored),
            "pooled_rank_ic": tsk["pooled_rank_ic"],
            "pooled_rank_ic_ci95": tsk["pooled_rank_ic_ci95"],
            "rmse": tsk["rmse"],
            "mae": tsk["mae"],
            "net_sharpe_by_cost_bps": tiers,
            "net_sharpe_base_ci95": tsk["net_sharpe_base_ci95"],
            "portfolio_returns": port_rets.tolist(),
            "portfolio_picks": portfolio["picks"].tolist(),
        }
        if name == "frozen_purged":
            exports = Path(cfg["paths"]["csv_exports"])
            scored.to_csv(exports / "oos_scores_frozen.csv", index=False)
            model.save(Path(cfg["paths"]["models"]) / "panel_anfis_frozen.json")

    if not args.skip_scan:
        w = report["frozen_scan_winner"]
        report["frozen_selection_confirmed"] = (
            int(w["n_rules"]) == FROZEN_SPEC["n_rules"]
            and float(w["cluster_radius"]) == FROZEN_SPEC["cluster_radius"]
            and float(w["ridge_lambda"]) == FROZEN_SPEC["ridge_lambda"])

    logs.mkdir(parents=True, exist_ok=True)
    (logs / "frozen_oos.json").write_text(
        json.dumps(_strict(report), indent=2, default=str), encoding="utf-8")

    for name in ("frozen_purged", "deployed_purged"):
        r = report[name]
        ci = r["monthly_rank_ic_bootstrap_ci95"]
        print(f"{name}: monthly IC {r['monthly_rank_ic']:+.3f} "
              f"(SE {r['monthly_rank_ic_se']:.3f}, boot CI "
              f"[{ci[0]:+.3f}, {ci[1]:+.3f}]), pooled IC "
              f"{r['pooled_rank_ic']:+.3f}, Sharpe@15bps "
              f"{r['net_sharpe_by_cost_bps']['15']:.3f}, @50 "
              f"{r['net_sharpe_by_cost_bps']['50']:.3f}, @150 "
              f"{r['net_sharpe_by_cost_bps']['150']:.3f}")
    print(f"\nfull report: {logs / 'frozen_oos.json'}")


def _date_clustered_bootstrap_ic(scored: pd.DataFrame, b: int = 10000,
                                 seed: int = 7) -> list[float]:
    """Percentile CI for the monthly cross-sectional Rank IC by resampling
    the rebalance dates (the natural cluster unit) with replacement."""
    _, _, per_month = _monthly_ic(scored)
    values = per_month.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(b, len(values)))].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return [float(low), float(high)]


def _monthly_ic(scored: pd.DataFrame):
    from src.backtest import monthly_rank_ic
    return monthly_rank_ic(scored)


def _strict(obj):
    """Recursively map NaN/inf floats to None so the JSON report is strict."""
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _strict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_strict(v) for v in obj]
    return obj


if __name__ == "__main__":
    main()
