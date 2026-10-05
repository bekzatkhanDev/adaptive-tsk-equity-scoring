"""Evaluation metrics (Section 4 of the paper).

Implements the out-of-sample statistics reported in the article:

* **Rank IC** -- Spearman rank correlation between scores and forward returns,
  per asset daily (overlapping), pooled asset-day, and monthly non-overlapping;
* **block bootstrap CIs** -- ``B = 5000`` draws of 30-day panel blocks;
* **net Sharpe** -- monthly top-N long portfolio with round-trip cost tiers
  (Section 4.3.1).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

__all__ = [
    "rank_ic",
    "per_asset_rank_ic",
    "pooled_rank_ic",
    "monthly_rank_ic",
    "block_bootstrap_ci",
    "top_n_spread_returns",
    "top_n_portfolio_returns",
    "net_sharpe",
    "deflated_sharpe_ratio",
    "period_marks",
    "mark_rank_ic",
]

# Period aliases for the Tier-1 quarterly/annual "mark" statistics.
_PERIOD_ALIAS = {"weekly": "W", "monthly": "M", "quarterly": "Q", "annual": "Y"}
_DEFAULT_MARK_HORIZON = {"W": 5, "M": 21, "Q": 63, "Y": 252}



def _rank(series: np.ndarray) -> np.ndarray:
    return stats.rankdata(series)


def rank_ic(scores: np.ndarray, returns: np.ndarray) -> float:
    """Spearman Rank IC on paired finite observations (NaN if fewer than 3)."""
    scores = np.asarray(scores, dtype=float)
    returns = np.asarray(returns, dtype=float)
    keep = np.isfinite(scores) & np.isfinite(returns)
    if keep.sum() < 3:
        return float("nan")
    correlation, _ = stats.spearmanr(scores[keep], returns[keep])
    return float(correlation)


def per_asset_rank_ic(panel: pd.DataFrame, score_col: str = "Y_attr",
                      ret_col: str = "R30") -> pd.Series:
    """Daily overlapping Rank IC within each ticker's time series."""
    return panel.groupby("ticker").apply(
        lambda g: rank_ic(g[score_col].to_numpy(), g[ret_col].to_numpy()),
        include_groups=False,
    )


def pooled_rank_ic(panel: pd.DataFrame, score_col: str = "Y_attr",
                   ret_col: str = "R30") -> float:
    """Rank ordering across all pooled asset-day observations."""
    return rank_ic(panel[score_col].to_numpy(), panel[ret_col].to_numpy())


def monthly_rank_ic(panel: pd.DataFrame, score_col: str = "Y_attr",
                    ret_col: str = "R30") -> tuple[float, float, pd.Series]:
    """Cross-sectional monthly Rank IC on the first trading day of each month.

    Returns ``(mean_ic, standard_error, per_month_series)``.
    """
    per_month: dict[pd.Timestamp, float] = {}
    for stamp, group in panel.groupby(pd.Grouper(key="date", freq="MS")):
        first_day = group["date"].min()
        day_rows = panel[panel["date"] == first_day]
        if day_rows["ticker"].nunique() >= 3:
            per_month[first_day] = rank_ic(
                day_rows[score_col].to_numpy(), day_rows[ret_col].to_numpy()
            )
    series = pd.Series(per_month, name="monthly_rank_ic").dropna()
    if series.empty:
        return float("nan"), float("nan"), series
    return float(series.mean()), float(series.std(ddof=1) / np.sqrt(len(series))), series


def block_bootstrap_ci(values: np.ndarray, b: int = 5000, block: int = 30,
                       seed: int = 0, alpha: float = 0.05) -> tuple[float, float, float]:
    """Moving-block bootstrap CI for the mean of a statistic series."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n < block:
        return float(values.mean()) if n else float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n - block + 1, size=(b, n // block + 1))
    means = np.empty(b)
    for i in range(b):
        pieces = [values[s:s + block] for s in starts[i]]
        means[i] = np.concatenate(pieces)[:n].mean()
    low, high = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(values.mean()), float(low), float(high)


def top_n_portfolio_returns(
    panel: pd.DataFrame,
    top_n: int = 3,
    score_col: str = "Y_attr",
    ret_col: str = "R30",
) -> pd.DataFrame:
    """Equal-weighted monthly top-N long portfolio (Section 4.3.1).

    Selects the top-N tickers by score on the first trading day of each month
    and earns their forward returns for that month.
    """
    rows = []
    for stamp, group in panel.groupby(pd.Grouper(key="date", freq="MS")):
        first_day = group["date"].min()
        day_rows = panel[panel["date"] == first_day]
        day_rows = day_rows[day_rows[ret_col].notna()]
        picks = day_rows.nlargest(top_n, score_col)
        if picks["ticker"].nunique() < top_n:
            continue
        rows.append({
            "date": first_day,
            "portfolio_return": float(picks[ret_col].mean()),
            "n_picks": int(len(picks)),
            "picks": ",".join(sorted(picks["ticker"])),
        })
    return pd.DataFrame(rows)


def net_sharpe(portfolio: pd.DataFrame, cost_bps: float = 15.0,
               turnover_annual: float | None = None) -> float:
    """Annualised net Sharpe of the monthly portfolio with round-trip costs.

    ``cost_bps`` is charged per round trip; if ``turnover_annual`` is not given
    it is estimated from realised month-over-month membership changes.
    """
    returns = portfolio["portfolio_return"].to_numpy(dtype=float)
    if turnover_annual is None:
        membership = [set(p.split(",")) for p in portfolio["picks"]]
        changes = [
            len(membership[i] ^ membership[i - 1]) / max(len(membership[i]), 1)
            for i in range(1, len(membership))
        ]
        monthly_turnover = float(np.mean(changes)) / 2.0 if changes else 0.0
        turnover_annual = monthly_turnover * 12.0
    monthly_cost = turnover_annual / 12.0 * cost_bps / 1e4
    excess = returns - monthly_cost
    std = float(np.std(excess, ddof=1))
    return float(np.mean(excess) / std * np.sqrt(12.0)) if std > 0 else float("nan")


def top_n_spread_returns(
    panel: pd.DataFrame,
    top_n: int = 3,
    score_col: str = "Y_attr",
    ret_col: str = "R30",
) -> pd.DataFrame:
    """Dollar-neutral monthly long-top-N / short-bottom-N spread portfolio.

    A cross-sectional ranking claim should be tested on a *spread*, not only on
    the long leg: with seven names and one test year, a long-only top-3 Sharpe
    is dominated by the market beta of the picks. Here each month's N highest
    scored names are funded long against the N lowest scored names (equal
    weight, unit gross exposure per side), so the reported series is close to
    market-neutral and measures the ordering information in the score itself.
    Costs are applied to both legs via ``net_sharpe``'s turnover estimate.
    """
    rows = []
    for stamp, group in panel.groupby(pd.Grouper(key="date", freq="MS")):
        first_day = group["date"].min()
        day_rows = panel[panel["date"] == first_day]
        day_rows = day_rows[day_rows[ret_col].notna()]
        if day_rows["ticker"].nunique() < 2 * top_n:
            continue
        longs = day_rows.nlargest(top_n, score_col)
        shorts = day_rows.nsmallest(top_n, score_col)
        rows.append({
            "date": first_day,
            "portfolio_return": float(longs[ret_col].mean() - shorts[ret_col].mean()),
            "n_picks": int(len(longs) + len(shorts)),
            "picks": ",".join(sorted(longs["ticker"])) + "|" +
                     ",".join(sorted(shorts["ticker"])),
        })
    return pd.DataFrame(rows)


def deflated_sharpe_ratio(returns: np.ndarray, n_trials: int,
                          periodicity: int = 12) -> dict[str, float]:
    """Deflated Sharpe ratio (Bailey & Lopez de Prado, 2014).

    Corrects the expected maximum Sharpe ratio for selection over ``n_trials``
    strategy variants (feature sets, hyper-parameter grids ...) and tests the
    observed Sharpe against that benchmark with a one-sided normal critical
    value, using the sample's skewness and excess kurtosis.

    Returns ``dsr`` (probability the true Sharpe exceeds the selection-adjusted
    benchmark), ``sharpe_obs``, ``sr_star`` (the deflated threshold) and
    ``t_stat``. ``periodicity`` annualises the frequency of ``returns``
    (12 for monthly, 250 for daily).
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    T = len(r)
    if T < 4 or n_trials < 1:
        return {"dsr": float("nan"), "sharpe_obs": float("nan"),
                "sr_star": float("nan"), "t_stat": float("nan")}
    mean, sd = float(np.mean(r)), float(np.std(r, ddof=1))
    if sd <= 0:
        return {"dsr": float("nan"), "sharpe_obs": float("nan"),
                "sr_star": float("nan"), "t_stat": float("nan")}
    sharpe = mean / sd
    skew = float(stats.skew(r, bias=False))
    kurt = float(stats.kurtosis(r, fisher=True))  # excess kurtosis
    euler = 0.5772156649015329
    if n_trials > 1:
        sr_max = (np.sqrt(2.0 * np.log(n_trials))
                  - (np.log(np.pi) + euler) / np.sqrt(2.0 * np.log(n_trials)))
    else:
        sr_max = 0.0
    sr_star = (sr_max * np.sqrt(1.0 - skew * sharpe
                                + ((kurt + 2.0) / 4.0) * sharpe ** 2)
               + 3.0 * skew / (4.0 * np.sqrt(float(T))))
    t_stat = (sharpe - sr_star) * np.sqrt(float(T)
                                          / (1.0 - skew * sharpe
                                             + ((kurt + 2.0) / 4.0) * sharpe ** 2))
    dsr = float(stats.norm.cdf(t_stat))
    return {"dsr": dsr, "sharpe_obs": sharpe * np.sqrt(periodicity),
            "sr_star": sr_star * np.sqrt(periodicity), "t_stat": t_stat}


def period_marks(
    panel: pd.DataFrame,
    freq: str = "M",
    agg: str = "mean",
    score_col: str = "Y_attr",
    ewma_span: int = 21,
) -> pd.DataFrame:
    """Per ``(ticker, period)`` mark = aggregate of the daily scores over the period.

    ``freq`` accepts pandas period aliases (``'W','M','Q','Y'``) or the words
    ``'weekly'/'monthly'/'quarterly'/'annual'``. ``agg`` is one of ``'mean'``,
    ``'median'``, ``'last'`` or ``'ewma'`` (exponentially weighted, ``ewma_span``).
    Returns a tidy frame ``[ticker, period, mark]`` where ``period`` is a pandas
    ``Period`` -- the "quarterly/annual mark" a corporate treasury board would see.
    """
    freq_key = _PERIOD_ALIAS.get(str(freq).lower(), str(freq))
    frame = panel[["ticker", "date", score_col]].copy()
    frame["period"] = frame["date"].dt.to_period(freq_key)
    grouped = frame.groupby(["ticker", "period"])[score_col]
    if agg == "mean":
        mark = grouped.mean()
    elif agg == "median":
        mark = grouped.median()
    elif agg == "last":
        mark = grouped.last()
    elif agg == "ewma":
        mark = grouped.apply(lambda s: s.ewm(span=ewma_span, adjust=False).mean().iloc[-1])
    else:
        raise ValueError("agg must be one of 'mean', 'median', 'last', 'ewma'")
    return mark.rename("mark").reset_index()


def mark_rank_ic(
    panel: pd.DataFrame,
    freq: str = "Q",
    horizon: int | None = None,
    agg: str = "mean",
    score_col: str = "Y_attr",
) -> tuple[float, float, pd.Series]:
    """Cross-sectional Rank IC of trailing period marks vs the next period's return.

    At each period boundary the cross-section of equities is ranked by the period
    *mark* (aggregated daily score over that period) and correlated with the
    forward ``R{horizon}`` recorded on the first trading day of the *following*
    period -- i.e. the mark is trailing and the return is forward, so the two
    windows never overlap. Defaults match the mark to its period length
    (weekly 5, monthly 21, quarterly 63, annual 252 trading days). Returns
    ``(mean_ic, standard_error, per_period_series)``.
    """
    freq_key = _PERIOD_ALIAS.get(str(freq).lower(), str(freq))
    horizon = int(horizon or _DEFAULT_MARK_HORIZON.get(freq_key, 21))
    ret_col = f"R{horizon}"
    if ret_col not in panel.columns:
        raise KeyError(f"mark_rank_ic needs '{ret_col}' in the panel")

    marks = period_marks(panel, freq=freq_key, agg=agg, score_col=score_col)

    forward = panel[["ticker", "date", ret_col]].dropna().copy()
    forward["period"] = forward["date"].dt.to_period(freq_key)
    first = (
        forward.sort_values("date")
        .groupby(["ticker", "period"], as_index=False)
        .first()[["ticker", "period", ret_col]]
        .rename(columns={ret_col: "fwd"})
    )
    # A mark formed *within* period p predicts the return earned *from* period p+1.
    first["period"] = first["period"] - 1

    merged = marks.merge(first, on=["ticker", "period"], how="inner")
    per_period: dict[pd.Timestamp, float] = {}
    for period, group in merged.groupby("period"):
        if group["ticker"].nunique() >= 3:
            per_period[period.to_timestamp()] = rank_ic(
                group["mark"].to_numpy(dtype=float), group["fwd"].to_numpy(dtype=float)
            )
    series = pd.Series(per_period, name="mark_rank_ic").dropna()
    if series.empty:
        return float("nan"), float("nan"), series
    return float(series.mean()), float(series.std(ddof=1) / np.sqrt(len(series))), series
