"""Portfolio inference (Referee Report, Requested Analysis 5).

With 7 names and 11 rebalances the realised top-3 Sharpe needs an explicit
null. This script computes:

* **permutation p-value** -- each month admits exactly C(7,3) = 35 equally
  likely portfolios under the no-skill null; Monte-Carlo draws one subset per
  month (B = 10000) and recomputes the net Sharpe, giving
  ``p = Pr(Sharpe_null >= Sharpe_realised)``. The exact per-month percentile
  of the realised subset among its 35 candidates is reported alongside;
* **leave-one-name-out** -- monthly IC and net Sharpe with each ticker dropped
  (does KZAP drive the result?);
* **equal-weight all-7 benchmark** -- the no-selection baseline the referee
  asks for;
* **paired TSK-vs-OLS test** -- monthly paired differences of the realised
  top-3 returns against the OLS baseline picks (paired t + Wilcoxon).

Dividends are loaded from ``data/raw/financials/dividends.csv`` and added to
forward labels for rows after a paid ex-date.  The supplied KZTK 2025 date is
the issuer's 25-May record date (a Sunday), so it should not be described as a
verified exchange ex-date without a corrected source field.

Example
-------
python experiments/portfolio_inference.py
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy import stats

from src.backtest import monthly_rank_ic, net_sharpe, top_n_portfolio_returns
from src.data_loader import FEATURE_COLUMNS, build_panel, load_config
from src.TSK_engine import TSKFuzzySystem
from experiments.evaluate_2025_oos import ols_baseline


def month_table(panel: pd.DataFrame, ret_col: str = "R30") -> list[dict]:
    """Per month: date, all-7 mean return, and the 35 subset returns."""
    out = []
    for _stamp, group in panel.groupby(pd.Grouper(key="date", freq="MS")):
        first_day = group["date"].min()
        day = panel[(panel["date"] == first_day) & panel[ret_col].notna()]
        tickers = sorted(day["ticker"].unique())
        if len(tickers) < 7:
            continue
        rets = {t: float(day.loc[day["ticker"] == t, ret_col].iloc[0])
                for t in tickers}
        subsets = list(itertools.combinations(tickers, 3))
        out.append({
            "date": str(first_day.date()),
            "tickers": tickers,
            "returns": rets,
            "subsets": {",".join(s): float(np.mean([rets[t] for t in s]))
                        for s in subsets},
        })
    return out


def permutation_test(panel: pd.DataFrame, b: int, seed: int,
                     cost_bps: float, months: list[dict]) -> dict:
    """Monte-Carlo no-skill null: one uniform random 3-subset per month."""
    realised = top_n_portfolio_returns(panel, top_n=3)
    sharpe_real = net_sharpe(realised, cost_bps=cost_bps)
    realised_picks = dict(zip(
        [m["date"] for m in months],
        [set(p.split(",")) for p in realised["picks"]]))

    # Exact per-month percentile of the realised subset among its 35.
    percentiles = []
    for m in months:
        got = realised_picks.get(m["date"])
        if not got:
            continue
        r = m["subsets"][",".join(sorted(got))]
        all_r = np.array(list(m["subsets"].values()))
        percentiles.append(float((all_r < r).mean()))

    rng = np.random.default_rng(seed)
    null = np.empty(b)
    for i in range(b):
        rets = []
        picks = []
        for m in months:
            key = rng.choice(list(m["subsets"].keys()))
            rets.append(m["subsets"][key])
            picks.append(key)
        # turnover-aware Sharpe on the synthetic ledger
        monthly_change = 0.0
        sets = [set(k.split(",")) for k in picks]
        changes = [len(sets[j] ^ sets[j - 1]) / 3.0 for j in range(1, len(sets))]
        monthly_change = float(np.mean(changes)) / 2.0 if changes else 0.0
        cost = monthly_change * 12.0 / 12.0 * cost_bps / 1e4
        excess = np.array(rets) - cost
        sd = float(np.std(excess, ddof=1))
        null[i] = np.mean(excess) / sd * np.sqrt(12.0) if sd > 0 else np.nan
    null = null[np.isfinite(null)]
    p = float((null >= sharpe_real).mean())
    return {
        "sharpe_realised": float(sharpe_real),
        "null_mean": float(null.mean()),
        "null_sd": float(null.std(ddof=1)),
        "p_one_sided": p,
        "b": int(len(null)),
        "realised_monthly_percentile_mean": float(np.mean(percentiles)),
        "realised_monthly_percentiles": percentiles,
    }


def leave_one_out(panel: pd.DataFrame, cost_bps: float) -> dict:
    """Monthly IC and net Sharpe with each ticker dropped in turn."""
    out = {}
    for dropped in sorted(panel["ticker"].unique()):
        sub = panel[panel["ticker"] != dropped]
        ic = monthly_rank_ic(sub)[0]
        port = top_n_portfolio_returns(sub, top_n=3)
        out[dropped] = {
            "monthly_ic": float(ic),
            "net_sharpe": float(net_sharpe(port, cost_bps=cost_bps)),
        }
    return out


def equal_weight_benchmark(panel: pd.DataFrame, cost_bps: float) -> dict:
    """All-7 equal-weight monthly portfolio (no selection at all)."""
    rows = []
    for _stamp, group in panel.groupby(pd.Grouper(key="date", freq="MS")):
        first_day = group["date"].min()
        day = panel[(panel["date"] == first_day) & panel["R30"].notna()]
        if day["ticker"].nunique() < 7:
            continue
        rows.append({"date": first_day,
                     "portfolio_return": float(day["R30"].mean()),
                     "picks": ",".join(sorted(day["ticker"]))})
    port = pd.DataFrame(rows)
    return {"net_sharpe": float(net_sharpe(port, cost_bps=cost_bps)),
            "mean_monthly_return": float(port["portfolio_return"].mean()),
            "n_months": int(len(port))}


def paired_tsk_vs_ols(panel: pd.DataFrame, model, cost_bps: float,
                      train: pd.DataFrame) -> dict:
    """Paired monthly test of realised TSK top-3 vs OLS-baseline top-3."""
    ols_scored, _ = ols_baseline(train, panel)
    tsk_port = top_n_portfolio_returns(panel, top_n=3)
    ols_port = top_n_portfolio_returns(ols_scored, top_n=3)
    merged = tsk_port.merge(ols_port, on="date", suffixes=("_tsk", "_ols"))
    diff = merged["portfolio_return_tsk"] - merged["portfolio_return_ols"]
    t_stat, t_p = stats.ttest_rel(
        merged["portfolio_return_tsk"], merged["portfolio_return_ols"])
    w_stat, w_p = stats.wilcoxon(diff)
    return {
        "n_months": int(len(merged)),
        "mean_diff": float(diff.mean()),
        "paired_t": float(t_stat), "paired_t_p": float(t_p),
        "wilcoxon_w": float(w_stat), "wilcoxon_p": float(w_p),
        "tsk_sharpe": float(net_sharpe(tsk_port, cost_bps=cost_bps)),
        "ols_sharpe": float(net_sharpe(ols_port, cost_bps=cost_bps)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="portfolio inference (analysis 5)")
    ap.add_argument("--b", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    cfg = load_config()
    model = TSKFuzzySystem.load(Path(cfg["paths"]["models"]) / "panel_anfis.json")
    train = build_panel(cfg, split="train")
    test = build_panel(cfg, split="test")
    test = test.assign(Y_attr=model.score(
        test[list(FEATURE_COLUMNS)].to_numpy(dtype=float)))
    cost = float(cfg["backtest"]["cost_tiers_bps"][0])

    from src.data_loader import load_dividends
    divs = load_dividends(cfg)

    months = month_table(test)
    report = {
        "n_months": len(months),
        "subsets_per_month": 35,
        "dividends_file_present": bool(len(divs)),
        "return_note": "price-only R30 (dividends.csv absent)" if not len(divs)
                       else "dividend data present",
        "permutation": permutation_test(test, args.b, args.seed, cost, months),
        "equal_weight_all7": equal_weight_benchmark(test, cost),
        "leave_one_out": leave_one_out(test, cost),
        "paired_tsk_vs_ols": paired_tsk_vs_ols(test, model, cost, train),
    }

    logs = Path(cfg["paths"]["logs"])
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "portfolio_inference.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")

    perm = report["permutation"]
    print(f"months={report['n_months']}  {report['return_note']}")
    print(f"realised Sharpe {perm['sharpe_realised']:.3f} vs null "
          f"{perm['null_mean']:.3f}+/-{perm['null_sd']:.3f} -> "
          f"p_one_sided={perm['p_one_sided']:.4f}")
    print(f"mean monthly percentile of realised subset: "
          f"{perm['realised_monthly_percentile_mean']:.2f}")
    ew = report["equal_weight_all7"]
    print(f"equal-weight all-7: Sharpe {ew['net_sharpe']:.3f} "
          f"mean_mret {ew['mean_monthly_return']:+.4f}")
    print("leave-one-out:")
    for k, v in report["leave_one_out"].items():
        print(f"  drop {k}: ic={v['monthly_ic']:.3f} sharpe={v['net_sharpe']:.3f}")
    pair = report["paired_tsk_vs_ols"]
    print(f"paired TSK vs OLS: diff={pair['mean_diff']:+.4f} "
          f"t_p={pair['paired_t_p']:.3f} wilcoxon_p={pair['wilcoxon_p']:.3f} "
          f"(tsk {pair['tsk_sharpe']:.3f} vs ols {pair['ols_sharpe']:.3f})")
    print(f"\nfull report: {logs / 'portfolio_inference.json'}")


if __name__ == "__main__":
    main()
