"""Out-of-sample evaluation on the test window (Section 4 of the paper).

Computes per-asset daily (overlapping) Rank IC, pooled asset-day Rank IC with a
30-day panel block-bootstrap CI, monthly non-overlapping Rank IC, and the
top-N monthly portfolio net Sharpe across cost tiers. Exports per-ticker OOS
CSVs and the paper figure dataset (``day, Y_attr, R30_scaled``).

Example
-------
python experiments/evaluate_2025_oos.py --ticker KMGZ --publish-paper-csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy import stats

from src.backtest import (
    monthly_rank_ic,
    net_sharpe,
    per_asset_rank_ic,
    pooled_rank_ic,
    rank_ic,
    top_n_portfolio_returns,
)
from src.data_loader import build_panel, feature_columns, load_config
from src.masking import masked_scores, prop1_bias_bound
from src.TSK_engine import TSKFuzzySystem

# Config-aware active input vector (canonical three inputs by default).
FEATURE_COLUMNS = feature_columns(load_config())


def _panel_block_bootstrap_ic(panel: pd.DataFrame, b: int, block: int, seed: int):
    """Pooled Rank IC CI via 30-day blocks over sorted unique dates."""
    dates = np.array(sorted(panel["date"].unique()))
    rows_by_date = {d: g for d, g in panel.groupby("date")}
    n_blocks_needed = int(np.ceil(len(dates) / block))
    rng = np.random.default_rng(seed)
    stats_boot = np.empty(b)
    for i in range(b):
        starts = rng.integers(0, len(dates) - block + 1, size=n_blocks_needed)
        picked = np.concatenate([dates[s:s + block] for s in starts])[: len(dates)]
        sample = pd.concat([rows_by_date[d] for d in picked], ignore_index=True)
        stats_boot[i] = pooled_rank_ic(sample)
    low, high = np.quantile(stats_boot, [0.025, 0.975])
    return float(low), float(high)


def _sharpe_bootstrap_ci(portfolio: pd.DataFrame, cost_bps: float, b: int, seed: int):
    """Net Sharpe CI by resampling calendar months with replacement."""
    rng = np.random.default_rng(seed)
    n = len(portfolio)
    sharpes = np.empty(b)
    for i in range(b):
        idx = rng.integers(0, n, size=n)
        sharpes[i] = net_sharpe(portfolio.iloc[idx].reset_index(drop=True), cost_bps)
    low, high = np.nanquantile(sharpes, [0.025, 0.975])
    return float(low), float(high)


def fit_metrics(scored: pd.DataFrame) -> dict:
    """RMSE / MAE against the bounded target and pooled Rank IC vs forward returns."""
    labelled = scored.dropna(subset=["Y_target"])
    prediction = labelled["Y_attr"].to_numpy(dtype=float)
    target = labelled["Y_target"].to_numpy(dtype=float)
    return {
        "n_labelled": int(len(labelled)),
        "rmse": float(np.sqrt(np.mean((prediction - target) ** 2))),
        "mae": float(np.mean(np.abs(prediction - target))),
        "pooled_rank_ic": pooled_rank_ic(scored),
    }


def ols_baseline(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Closed-form OLS on identical features/targets (Table 2 baseline)."""
    X = train[list(FEATURE_COLUMNS)].to_numpy(dtype=float)
    y = train["Y_target"].to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    scored = test.assign(
        Y_attr=beta[0] + test[list(FEATURE_COLUMNS)].to_numpy(dtype=float) @ beta[1:]
    )
    return scored, beta


def conditioning_report(model: TSKFuzzySystem, train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Check Proposition 1's constant ``C = max|y_k - Y_raw|`` on train and test.

    ``C`` is evaluated on the *deployed* path, i.e. with the rule outputs bounded
    to the score domain exactly as :meth:`TSKFuzzySystem.score` does, so the
    reported bound is the one that applies to the system whose scores are used.
    """

    def _stats(panel: pd.DataFrame) -> tuple[float, float, float]:
        X = panel[list(FEATURE_COLUMNS)].to_numpy(dtype=float)
        weights = model._normalize(model._log_firing(X))
        raw_rules = model.rule_outputs(X)
        if model.clip_rule_outputs:
            raw_rules = np.clip(raw_rules, 0.0, model.score_scale)
        aggregate = np.sum(weights * raw_rules, axis=1)
        return (
            float(np.abs(raw_rules).max()),
            float(np.abs(raw_rules - aggregate[:, None]).max()),
            float(np.abs(raw_rules).mean()),
        )

    train_max, train_c, train_mean = _stats(train)
    test_max, test_c, test_mean = _stats(test)
    return {
        "clip_rule_outputs": bool(model.clip_rule_outputs),
        "train": {
            "rule_output_max_abs": train_max,
            "rule_output_mean_abs": train_mean,
            "empirical_C": train_c,
            "bias_bound_with_realised_C": prop1_bias_bound(c=train_c),
        },
        "test": {
            "rule_output_max_abs": test_max,
            "rule_output_mean_abs": test_mean,
            "empirical_C": test_c,
            "bias_bound_with_realised_C": prop1_bias_bound(c=test_c),
        },
        "prop1_bound_nominal_C100": prop1_bias_bound(),
        "prop1_assumption_holds": bool(train_c <= 100.0 and test_c <= 100.0),
    }


def masked_path_report(
    cfg: dict, model: TSKFuzzySystem, panel: pd.DataFrame, draws: int = 25
) -> dict:
    """Deployed-path metrics with Pre-AGM masking armed (Section 3.4).

    Scores Pre-AGM rows through the masked rule weights and leaves the rest
    untouched, then reports the resulting skill, the realised utility cost and
    the empirical bias against Proposition 1's bound, averaged over ``draws``
    independent masking draws.
    """
    alpha = float(cfg["masking"]["alpha"])
    X = panel[list(FEATURE_COLUMNS)].to_numpy(dtype=float)
    base = model.score(X)
    flags = panel["pre_agm"].to_numpy(dtype=float)
    in_window = flags > 0.0
    labelled = panel["Y_target"].notna().to_numpy()

    pooled_ics, rmses, sharpes, abs_deltas, biases = [], [], [], [], []
    for seed in range(draws):
        masked = masked_scores(model, X, flags, alpha=alpha, rng=np.random.default_rng(seed))
        delta = masked - base
        abs_deltas.append(float(np.abs(delta[in_window]).mean()))
        biases.append(float(delta[in_window].mean()))
        scored = panel.assign(Y_attr=masked)
        pooled_ics.append(pooled_rank_ic(scored))
        subset = scored[labelled]
        rmses.append(
            float(
                np.sqrt(
                    np.mean((subset["Y_attr"] - subset["Y_target"]) ** 2)
                )
            )
        )
        portfolio = top_n_portfolio_returns(scored, top_n=int(cfg["backtest"]["top_n"]))
        sharpes.append(net_sharpe(portfolio, cost_bps=cfg["backtest"]["cost_tiers_bps"][0]))

    return {
        "alpha": alpha,
        "draws": draws,
        "n_masked_rows": int(in_window.sum()),
        "mean_abs_delta_on_pre_agm": float(np.mean(abs_deltas)),
        "mean_signed_bias": float(np.mean(biases)),
        "mean_abs_bias": float(np.mean(np.abs(biases))),
        "pooled_rank_ic_mean": float(np.mean(pooled_ics)),
        "pooled_rank_ic_sd": float(np.std(pooled_ics, ddof=1)),
        "rmse_mean": float(np.mean(rmses)),
        "rmse_sd": float(np.std(rmses, ddof=1)),
        "net_sharpe_base_mean": float(np.mean(sharpes)),
        "net_sharpe_base_sd": float(np.std(sharpes, ddof=1)),
    }


def evaluate(cfg: dict, model: TSKFuzzySystem, train: pd.DataFrame, test: pd.DataFrame) -> dict:
    panel = test.assign(Y_attr=model.score(test[list(FEATURE_COLUMNS)].to_numpy(dtype=float)))

    daily = per_asset_rank_ic(panel)
    ci_low, ci_high = _panel_block_bootstrap_ic(
        panel, int(cfg["backtest"]["bootstrap_B"]), int(cfg["backtest"]["bootstrap_block"]), 0
    )
    mean_m, se_m, per_month = monthly_rank_ic(panel)

    portfolio = top_n_portfolio_returns(panel, top_n=int(cfg["backtest"]["top_n"]))
    tiers = {
        str(bps): net_sharpe(portfolio, cost_bps=bps)
        for bps in cfg["backtest"]["cost_tiers_bps"]
    }
    sharpe_low, sharpe_high = _sharpe_bootstrap_ci(
        portfolio, cost_bps=cfg["backtest"]["cost_tiers_bps"][0],
        b=min(int(cfg["backtest"]["bootstrap_B"]), 2000), seed=1,
    )

    # Baseline on identical features/targets.
    ols_scored, beta = ols_baseline(train, test)
    ols_portfolio = top_n_portfolio_returns(ols_scored, top_n=int(cfg["backtest"]["top_n"]))
    ols_tiers = {
        str(bps): net_sharpe(ols_portfolio, cost_bps=bps)
        for bps in cfg["backtest"]["cost_tiers_bps"]
    }

    return {
        "n_train_rows": int(len(train)),
        "n_train_days": int(train["date"].nunique()),
        "n_test_rows": int(len(panel)),
        "n_test_days": int(panel["date"].nunique()),
        "tsk": {
            **fit_metrics(panel),
            "n_rules": int(model.n_rules),
            "n_free_params": int(model.n_free_params),
            "n_total_params": int(model.n_total_params),
            "conditioning": conditioning_report(model, train, panel),
            "masked_path": masked_path_report(cfg, model, panel),
            "pooled_rank_ic_ci95": [ci_low, ci_high],
            "daily_rank_ic_per_asset": {
                k: (None if pd.isna(v) else round(v, 4)) for k, v in daily.items()
            },
            "daily_rank_ic_panel_mean": float(np.nanmean(daily)),
            "monthly_rank_ic_mean": mean_m,
            "monthly_rank_ic_se": se_m,
            "net_sharpe_by_cost_bps": tiers,
            "net_sharpe_base_ci95": [sharpe_low, sharpe_high],
        },
        "ols": {
            **fit_metrics(ols_scored),
            "n_params": int(len(beta)),
            "net_sharpe_by_cost_bps": ols_tiers,
        },
        "monthly_rank_ic_per_month": {str(k.date()): round(v, 4) for k, v in per_month.items()},
        "portfolio_picks": portfolio["picks"].tolist(),
        "ols_portfolio_picks": ols_portfolio["picks"].tolist(),
    }, panel, portfolio


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default=None, help="also export one ticker's OOS series")
    parser.add_argument("--model", default=None, help="panel model JSON (default: train_anfis_panel output)")
    parser.add_argument("--publish-paper-csv", action="store_true",
                        help="refresh paper/sn-article-template/kmgz_2025_out_of_sample.csv")
    args = parser.parse_args()

    cfg = load_config()
    model_path = Path(args.model or Path(cfg["paths"]["models"]) / "panel_anfis.json")
    if not model_path.exists():
        raise SystemExit(f"model not found at {model_path} -- run train_anfis_panel.py first")
    model = TSKFuzzySystem.load(model_path)

    train = build_panel(cfg, split="train").dropna(subset=["Y_target"])
    test = build_panel(cfg, split="test")
    report, scored, portfolio = evaluate(cfg, model, train, test)

    exports = Path(cfg["paths"]["csv_exports"])
    scored.to_csv(exports / "oos_scores.csv", index=False)
    portfolio.to_csv(exports / "oos_portfolio.csv", index=False)
    Path(cfg["paths"]["logs"]).joinpath("oos_2025_report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )

    if args.ticker:
        rows = scored[scored["ticker"] == args.ticker].reset_index(drop=True)
        out = pd.DataFrame({
            "date": rows["date"],
            "day": np.arange(1, len(rows) + 1),
            "Y_attr": rows["Y_attr"],
            "R30_scaled": 50.0 * (1.0 + np.tanh(0.3 * rows["Z30"].fillna(0.0))),
            "R30": rows["R30"],
            "pre_agm": rows["pre_agm"],
        })
        out.to_csv(exports / f"{args.ticker}_2025_oos.csv", index=False)
        if args.publish_paper_csv:
            paper_dir = Path(cfg["paths"]["paper"])
            out[["day", "Y_attr", "R30_scaled"]].to_csv(
                paper_dir / "kmgz_2025_out_of_sample.csv", index=False
            )

    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
