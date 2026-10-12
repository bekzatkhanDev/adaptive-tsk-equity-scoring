"""Purged splits + full-search deflated Sharpe (Referee Report, Analysis 6).

Two fixes the referee requests:

1. **Purge/embargo the train/test boundary.** R30 labels stamped in the last
   ``target.horizon`` trading days of 2024 are forward returns computed over
   2025 prices, so the canonical split leaks. This script drops every training
   row whose forward window crosses ``train_end`` (30-trading-day embargo on
   the global trading calendar), retrains the deployed configuration from
   scratch, and re-evaluates IC/Sharpe on 2025 against the unpurged baseline.
2. **DSR over the full search.** The paper deflates only the 20 Tier-2 trials.
   Here DSR is recomputed on the canonical portfolio return series across a
   range of trial counts covering the documented search: 36 (K x radius x
   ridge), x3 clv modes, x4 feature sets (=432 full product), so the reader
   can see the sensitivity instead of a single convenient number.

Example
-------
python experiments/purged_robustness.py
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
from src.backtest import (
    deflated_sharpe_ratio,
    monthly_rank_ic,
    net_sharpe,
    pooled_rank_ic,
    top_n_portfolio_returns,
)
from src.data_loader import FEATURE_COLUMNS, build_panel, load_config
from src.TSK_engine import TSKFuzzySystem


def purge_train(train: pd.DataFrame, full: pd.DataFrame, horizon: int):
    """Drop train rows whose forward ``horizon``-day window crosses train_end.

    Uses the global trading calendar (all unique dates across the pooled
    panel): a row is purged when its stamp sits within ``horizon`` trading
    days of the last training date, because that row's forward return is
    computed over prices from the test year.
    """
    dates = np.sort(pd.concat([train["date"], full["date"]]).unique())
    train_end = train["date"].max()
    pos = int(np.searchsorted(dates, train_end, side="right")) - 1
    cutoff = dates[max(pos - horizon, 0)]
    return train[train["date"] < cutoff].copy(), str(pd.Timestamp(cutoff).date())


def fit_model(cfg: dict, train: pd.DataFrame) -> TSKFuzzySystem:
    """Same architecture/hyper-parameters as the deployed configuration."""
    rows = train.dropna(subset=["Y_target"])
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
    trainer.fit(X, y, epochs=int(tsk_cfg["epochs"]), verbose_every=0)
    return model


def evaluate(model: TSKFuzzySystem, test: pd.DataFrame, cfg: dict) -> dict:
    X = test[list(FEATURE_COLUMNS)].to_numpy(dtype=float)
    scored = test.assign(Y_attr=model.score(X))
    port = top_n_portfolio_returns(scored, top_n=int(cfg["backtest"]["top_n"]))
    bps = float(cfg["backtest"]["cost_tiers_bps"][0])
    return {
        "pooled_rank_ic": float(pooled_rank_ic(scored)),
        "monthly_rank_ic": float(monthly_rank_ic(scored)[0]),
        "monthly_rank_ic_se": float(monthly_rank_ic(scored)[1]),
        "net_sharpe_15bps": float(net_sharpe(port, cost_bps=bps)),
        "portfolio_returns": port["portfolio_return"].tolist(),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="purged split + full-search DSR (analysis 6)")
    ap.add_argument("--horizon", type=int, default=None,
                    help="embargo in trading days (default: target.horizon)")
    ap.add_argument("--skip-retrain", action="store_true",
                    help="only DSR, no retraining")
    args = ap.parse_args()

    cfg = load_config()
    horizon = int(args.horizon or cfg["target"]["horizon"])
    logs = Path(cfg["paths"]["logs"])
    report: dict = {"embargo_trading_days": horizon}

    test = build_panel(cfg, split="test")

    if not args.skip_retrain:
        train = build_panel(cfg, split="train")
        full = pd.concat([train, test], ignore_index=True)
        purged, cutoff = purge_train(train, full, horizon)
        report.update({
            "n_train_original": int(len(train.dropna(subset=["Y_target"]))),
            "n_train_purged": int(len(purged.dropna(subset=["Y_target"]))),
            "purge_cutoff": cutoff,
        })
        # Baseline: deployed model (trained on the unpurged split).
        deployed = TSKFuzzySystem.load(
            Path(cfg["paths"]["models"]) / "panel_anfis.json")
        report["unpurged"] = evaluate(deployed, test, cfg)
        # Purged retrain, identical configuration.
        purged_model = fit_model(cfg, purged)
        report["purged"] = evaluate(purged_model, test, cfg)
        logs.mkdir(parents=True, exist_ok=True)
        purged_model.save(Path(cfg["paths"]["models"]) / "panel_anfis_purged.json")

    # Full-search DSR on the deployed portfolio return series.
    if "unpurged" in report:
        rets = np.array(report["unpurged"]["portfolio_returns"], dtype=float)
    else:
        deployed = TSKFuzzySystem.load(
            Path(cfg["paths"]["models"]) / "panel_anfis.json")
        rets = np.array(evaluate(deployed, test, cfg)["portfolio_returns"])
    counts = {"tsk_grid_36": 36, "plus_clv_x3": 108, "plus_feature_sets_x4": 432,
              "tier2_only_20": 20, "canonical_1": 1}
    report["dsr_by_trial_count"] = {
        name: deflated_sharpe_ratio(rets, n_trials=n, periodicity=12)
        for name, n in counts.items()
    }
    report["search_dimensions"] = (
        "K in {3,5,7} x radius in {0.3,0.4,0.5,0.6} x ridge in {1,10,100} "
        "= 36; clv_mode in {clv,tick,hybrid}; 4 feature sets; DSR reported "
        "at each nesting level rather than only the 20 Tier-2 trials")

    (logs / "purged_robustness.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")

    if "unpurged" in report:
        u, p = report["unpurged"], report["purged"]
        print(f"embargo {horizon} trading days -> cutoff {report['purge_cutoff']}; "
              f"train {report['n_train_original']} -> {report['n_train_purged']} rows")
        print(f"unpurged: ic_m={u['monthly_rank_ic']:.3f} sharpe={u['net_sharpe_15bps']:.3f}")
        print(f"purged:   ic_m={p['monthly_rank_ic']:.3f} sharpe={p['net_sharpe_15bps']:.3f}")
    for name, r in report["dsr_by_trial_count"].items():
        print(f"DSR[{name}]: {r['dsr']:.4f} (sr*={r['sr_star']:.3f}, "
              f"obs={r['sharpe_obs']:.3f})")
    print(f"\nfull report: {logs / 'purged_robustness.json'}")


if __name__ == "__main__":
    main()
