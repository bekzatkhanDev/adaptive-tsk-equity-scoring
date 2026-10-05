"""Data ingestion and Section-3.1 feature construction (no lookahead).

Raw formats handled (all confirmed against the real files in ``data/raw``):

* OHLC -- Investing.com exports: ``Date, Price, Open, High, Low, Vol., Change %``
  with US ``MM/DD/YYYY`` dates, thousands commas and volume suffixes (``K/M/B``).
* Financials -- one export per year (``financials-2024.csv``, ``financials-2025.csv``)
  with Russian headers, per-quarter (``Q1..Q4``) or half-year (``H1/H2``) EBITDA /
  Net Debt / FCFE in mln KZT, plus an annual ``За <year> год`` total row. The
  2024 export is semicolon-delimited and the 2025 export comma-delimited, so the
  delimiter is detected per file.
* Macro -- ``tonia_rbk.xlsx`` (daily TONIA close, ``DD.MM.YY``) and
  ``rate_nbk.xlsx`` (NBK base-rate step changes).

Every feature at time *t* uses the trailing window ``[t-N, t-1]`` only
(:func:`compute_features` implements Eqs. (1)-(4) of the paper); the forward
30-day risk-adjusted return of Eq. (7) is computed separately as the *label*.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "load_config",
    "load_ohlc",
    "load_all_ohlc",
    "financials_paths",
    "load_financials",
    "load_tonia",
    "load_fx",
    "load_dividends",
    "risk_free_daily",
    "compute_features",
    "compute_target",
    "target_horizons",
    "feature_columns",
    "build_fundamentals_features",
    "FUNDAMENTAL_COLUMNS",
    "train_window_zstats",
    "standardize_target",
    "build_panel",
    "FEATURE_COLUMNS",
]

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

_VOL_SUFFIX = {"K": 1e3, "M": 1e6, "B": 1e9}

_PERIOD_MONTH_END = {
    "Q1": (3, 31), "Q2": (6, 30), "Q3": (9, 30), "Q4": (12, 31),
    "H1": (6, 30), "H2": (12, 31),
}


def load_config(path=None) -> dict:
    """Read ``config.yaml`` (project root by default)."""
    import yaml

    cfg_path = Path(path) if path else _PROJECT_ROOT / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# --------------------------------------------------------------------- OHLC
def _parse_number(value) -> float:
    """``"14,654.70"`` -> 14654.7; empty/``-`` -> NaN."""
    if value is None:
        return np.nan
    text = str(value).strip().replace("\ufeff", "").replace(",", "")
    if text in {"", "-", "NA", "N/A", "nan"}:
        return np.nan
    try:
        return float(text)
    except ValueError:
        return np.nan


def _parse_volume(value) -> float:
    """``"1.43K"`` -> 1430.0; plain numbers pass through; empty -> NaN."""
    if value is None:
        return np.nan
    text = str(value).strip().replace(",", "")
    if text in {"", "-", "NA", "N/A", "nan"}:
        return np.nan
    match = re.fullmatch(r"([0-9.]+)\s*([KMB]?)", text, flags=re.IGNORECASE)
    if not match:
        return np.nan
    return float(match.group(1)) * _VOL_SUFFIX.get(match.group(2).upper(), 1.0)


def load_ohlc(path, ticker: str = "") -> pd.DataFrame:
    """Parse one Investing.com OHLCV export into a canonical ascending frame.

    Returns columns ``[date, ticker, open, high, low, close, volume]`` with
    ``date`` a normalised ``datetime64[ns]`` column, sorted ascending.
    """
    frame = pd.read_csv(path, encoding="utf-8-sig")
    frame.columns = [c.strip() for c in frame.columns]
    if "Vol." in frame.columns:
        volume = frame["Vol."]
    elif "Volume" in frame.columns:
        volume = frame["Volume"]
    else:
        volume = pd.Series(np.nan, index=frame.index)
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(frame["Date"], format="%m/%d/%Y", errors="coerce"),
            "ticker": ticker or Path(path).stem.upper(),
            "open": frame["Open"].map(_parse_number),
            "high": frame["High"].map(_parse_number),
            "low": frame["Low"].map(_parse_number),
            "close": frame["Price"].map(_parse_number),
            "volume": volume.map(_parse_volume),
        }
    )
    return out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


def load_all_ohlc(cfg: dict) -> dict[str, pd.DataFrame]:
    """Load every configured ticker; raises if a raw file is missing."""
    base = _PROJECT_ROOT / cfg["paths"]["raw_ohlc"]
    frames: dict[str, pd.DataFrame] = {}
    for ticker in cfg["data"]["tickers"]:
        path = base / cfg["data"]["ohlc_file_pattern"].format(ticker=ticker)
        frames[ticker] = load_ohlc(path, ticker)
    return frames


# --------------------------------------------------------------- financials
_FINANCIAL_RENAME = {
    "Тикер": "ticker",
    "Квартал / Полугодие": "period",
    "EBITDA (млн KZT)": "ebitda",
    "Net Debt / Чистый долг (млн KZT)": "net_debt",
    "FCFE / Свободный поток (млн KZT)": "fcfe",
}


def _detect_delimiter(path: Path) -> str:
    """Header sniff for the per-year exports.

    The exports are inconsistent by year: ``\\t`` for 2022/2023, ``;`` for 2024
    and ``,`` for 2025, so all three candidates are counted and the most frequent
    wins (ties fall back to ``,``).
    """
    header = path.read_text(encoding="utf-8-sig").splitlines()[0]
    counts = {candidate: header.count(candidate) for candidate in ("\t", ";", ",")}
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else ","


def _year_from_text(text) -> int | None:
    """Extract a 4-digit year from a period label such as ``За 2024 год``."""
    match = re.search(r"(20\d{2})", str(text))
    return int(match.group(1)) if match else None


def financials_paths(cfg: dict, years=None) -> list[tuple[int, Path]]:
    """Resolve ``(year, path)`` pairs for the per-year financials exports.

    ``years`` defaults to ``data.financials_years``. Missing years are skipped;
    a single legacy ``paths.raw_financials`` file is used as a fallback.
    """
    paths_cfg = cfg.get("paths", {})
    pattern = paths_cfg.get("raw_financials_pattern")
    wanted = years if years is not None else cfg.get("data", {}).get("financials_years")
    resolved: list[tuple[int, Path]] = []
    if pattern and wanted:
        for year in wanted:
            candidate = _PROJECT_ROOT / str(pattern).format(year=year)
            if candidate.exists():
                resolved.append((int(year), candidate))
    if not resolved:
        legacy = paths_cfg.get("raw_financials")
        if legacy and (_PROJECT_ROOT / legacy).exists():
            path = _PROJECT_ROOT / legacy
            year = _year_from_text(path.stem) or int(
                max(frame["date"].dt.year.max() for frame in load_all_ohlc(cfg).values())
            )
            resolved.append((year, path))
    if not resolved:
        raise FileNotFoundError(
            "no financials export found -- check paths.raw_financials_pattern "
            "and data.financials_years"
        )
    return resolved


def _read_financials_file(path: Path, year: int) -> pd.DataFrame:
    """Parse one per-year financials export into tidy rows."""
    raw = pd.read_csv(path, sep=_detect_delimiter(path), encoding="utf-8-sig")
    raw.columns = [c.strip() for c in raw.columns]
    missing = [c for c in _FINANCIAL_RENAME if c not in raw.columns]
    if missing:
        raise KeyError(f"{path.name}: unrecognised column(s) {missing}")
    out = raw.rename(columns=_FINANCIAL_RENAME)[
        ["ticker", "period", "ebitda", "net_debt", "fcfe"]
    ].copy()
    out["ticker"] = out["ticker"].astype(str).str.strip().str.upper()
    out["period"] = out["period"].astype(str).str.strip()
    for column in ("ebitda", "net_debt", "fcfe"):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    # Annual "За <year> год" totals share the year-end period_end with Q4/H2 and
    # are flagged so the payout mapping can prefer the full-year figures.
    out["is_annual"] = out["period"].str.upper().str.startswith("ЗА")
    out["row_year"] = (
        out["period"].map(_year_from_text).astype("Float64").fillna(float(year)).astype(int)
    )
    month_day = out["period"].str.upper().map(_PERIOD_MONTH_END)
    out["period_end"] = [
        pd.Timestamp(year=int(row_year), month=int(md[0]), day=int(md[1]))
        if isinstance(md, tuple)
        else pd.NaT
        for row_year, md in zip(out["row_year"], month_day)
    ]
    out.loc[out["is_annual"], "period_end"] = [
        pd.Timestamp(year=int(y), month=12, day=31) for y in out.loc[out["is_annual"], "row_year"]
    ]
    out["period_year"] = out["row_year"].astype(int)
    return out.drop(columns=["row_year"])


def load_financials(cfg: dict, years=None) -> pd.DataFrame:
    """Tidy point-in-time fundamentals pooled over all per-year exports.

    Columns: ``ticker, period, period_end, ebitda, net_debt, fcfe, is_annual,
    period_year`` (values in mln KZT), sorted by ``(ticker, period_end, is_annual)``
    so that the last row per ticker is its most recent -- and, where published,
    full-year -- disclosure.
    """
    frames = [_read_financials_file(path, year) for year, path in financials_paths(cfg, years)]
    out = pd.concat(frames, ignore_index=True)
    empty_end = int(out["period_end"].isna().sum())
    if empty_end:
        print(f"[data_loader] WARNING: {empty_end} financials row(s) with unrecognised period label")
    out = out.dropna(subset=["period_end"])
    return out.sort_values(["ticker", "period_end", "is_annual"]).reset_index(drop=True)


# --------------------------------------------------------------------- macro
def _excel_header_row(path: Path, marker: str, scan: int = 10) -> int:
    """Index of the first row containing ``marker`` (exports carry title rows)."""
    preview = pd.read_excel(path, sheet_name=0, header=None, nrows=scan)
    for index, row in preview.iterrows():
        if row.astype(str).str.strip().eq(marker).any():
            return int(index)
    raise KeyError(f"{Path(path).name}: header row containing {marker!r} not found")


def load_tonia(cfg: dict) -> pd.DataFrame:
    """Daily TONIA close (percent per annum) from the RBK export.

    The sheet carries a title row above the real header, and the sheet name
    embeds the export date, so the header is located by its ``Дата`` marker.
    """
    path = _PROJECT_ROOT / cfg["paths"]["raw_tonia"]
    header = _excel_header_row(path, "Дата")
    frame = pd.read_excel(path, sheet_name=0, header=header)
    frame.columns = [str(c).strip() for c in frame.columns]
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(frame["Дата"], format="%d.%m.%y", errors="coerce"),
            "tonia": pd.to_numeric(frame["Закрытие"], errors="coerce"),
        }
    )
    return out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


def load_fx(cfg: dict) -> pd.DataFrame:
    """Daily USD/KZT reference rate (NBK export).

    Contextual macro series: it is reported alongside the results but is *not*
    an input to the three-input TSK engine of Section 3.2.
    """
    path = _PROJECT_ROOT / cfg["paths"]["raw_fx"]
    frame = pd.read_excel(path, sheet_name=0)
    frame.columns = [str(c).strip() for c in frame.columns]
    date_col = frame.columns[0]
    rate_col = next((c for c in frame.columns if c.upper() == "USD"), frame.columns[-1])
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(frame[date_col], format="%d.%m.%Y", errors="coerce"),
            "usd_kzt": pd.to_numeric(frame[rate_col], errors="coerce"),
        }
    )
    return out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


def risk_free_daily(tonia: pd.DataFrame) -> pd.DataFrame:
    """Daily simple risk-free rate ``r_daily = TONIA / 100 / 252``."""
    out = tonia.copy()
    out["rf_daily"] = out["tonia"].astype(float) / 100.0 / 252.0
    return out


# ------------------------------------------------------------------ dividends
def load_dividends(cfg: dict) -> pd.DataFrame:
    """Optional dividends file ``ticker, ex_date, dividend`` (KZT per share).

    Returns an empty (correctly typed) frame when the file is absent.
    """
    path = _PROJECT_ROOT / cfg["paths"].get("raw_dividends", "")
    if not path or not Path(path).exists():
        return pd.DataFrame({"ticker": [], "ex_date": [], "dividend": []})
    frame = pd.read_csv(path, encoding="utf-8-sig")
    out = pd.DataFrame(
        {
            "ticker": frame["ticker"].astype(str).str.upper().str.strip(),
            "ex_date": pd.to_datetime(frame["ex_date"], errors="coerce"),
            "dividend": pd.to_numeric(frame["dividend"], errors="coerce"),
        }
    ).dropna(subset=["ex_date"])
    return out


# ------------------------------------------------------------------- features
def _rolling_vwap(close: pd.Series, volume: pd.Series, window: int) -> pd.Series:
    """``VWAP_{t-1}(N)``: trailing-N volume-weighted mean, excluding day *t*."""
    num = (close * volume).rolling(window, min_periods=window).sum()
    den = volume.rolling(window, min_periods=window).sum()
    return (num / den.replace(0.0, np.nan)).shift(1)


def _order_flow(frame: pd.DataFrame, mode: str, eps_clv: float) -> pd.Series:
    """CLV / tick-rule / hybrid directional pressure (Eq. (4) / Algorithm 1)."""
    close = frame["close"]
    high, low = frame["high"], frame["low"]
    clv = (2.0 * close - high - low) / (high - low + eps_clv)
    tick = np.sign(close - close.shift(1))
    if mode == "clv":
        return clv
    if mode == "tick":
        return tick
    if mode == "hybrid":
        valid = high.notna() & low.notna() & (high > low)
        return clv.where(valid, tick)
    raise ValueError("of_mode must be one of 'clv', 'tick', 'hybrid'")


def compute_features(ohlc: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Section-3.1 features for one ticker, strictly trailing (no lookahead).

    Columns: ``date, ticker, close, volume, dev_price, dev_volume, order_flow``.
    ``dev_price`` / ``dev_volume`` use only ``[t-N, t-1]``; ``order_flow`` is the
    day-*t* CLV (or tick/hybrid fallback) -- a same-day signal, not lookahead.
    """
    feats = cfg.get("features", {})
    window = int(feats.get("vwap_window", 20))
    vol_window = int(feats.get("volume_window", window))
    eps_rel = float(feats.get("epsilon_volume_rel", 1e-6))
    mode = str(feats.get("of_mode", "hybrid"))

    frame = ohlc.sort_values("date").reset_index(drop=True)
    close = frame["close"].astype(float)
    volume = frame["volume"].astype(float)

    vwap = _rolling_vwap(close, volume, window)
    vwap_60 = _rolling_vwap(close, volume, 60)
    mean_vol = volume.rolling(vol_window, min_periods=vol_window).mean().shift(1)
    eps = eps_rel * mean_vol
    median_vol = volume.rolling(vol_window, min_periods=vol_window).median().shift(1)

    # Tier-2 multi-scale microstructure: all strictly trailing (shift / shifted
    # VWAP baseline), never look-ahead. Computed unconditionally so the export and
    # the enrichment share one code path; only the configured columns enter the
    # TSK input vector (see ``feature_columns``).
    ret_1 = close.pct_change()
    clv_daily = _order_flow(frame, "clv", eps_rel)
    extra = {
        "dev_price_60": (close - vwap_60) / vwap_60 * 100.0,
        "ret_5": (close.shift(1) / close.shift(6) - 1.0) * 100.0,
        "vol_20": ret_1.rolling(20, min_periods=20).std().shift(1) * 100.0,
        "clv_5": clv_daily.rolling(5, min_periods=5).mean().shift(1),
        "amihud": (
            ret_1.abs() / (close * volume).replace(0.0, np.nan)
        ).rolling(20, min_periods=20).mean().shift(1) * 1e6,
    }

    out = pd.DataFrame(
        {
            "date": frame["date"],
            "ticker": frame["ticker"],
            "close": close,
            "volume": volume,
            "dev_price": (close - vwap) / vwap * 100.0,
            "dev_volume": np.log((volume + eps) / (median_vol + eps)),
            # Eq. (4) reuses the same epsilon as Eq. (2); it is only a
            # division-by-zero guard because hybrid mode requires H_t > L_t.
            "order_flow": _order_flow(frame, mode, eps_rel),
            **extra,
        }
    )
    return out.replace([np.inf, -np.inf], np.nan)


# Multi-horizon grid (Tier-1). ``target.horizon`` is always included on top of
# ``target.horizons`` so the canonical R30 / Y_target are never lost.
_DEFAULT_HORIZONS = (5, 21, 63, 252)


def target_horizons(cfg: dict) -> list[int]:
    """Sorted trading-day horizons for the multi-horizon labels and marks.

    Equals ``sorted(set(target.horizons) | {target.horizon})``; with the shipped
    config this is ``[5, 21, 30, 63, 252]``.
    """
    target_cfg = cfg.get("target", {})
    primary = int(target_cfg.get("horizon", 30))
    extra = target_cfg.get("horizons") or list(_DEFAULT_HORIZONS)
    return sorted({int(h) for h in extra} | {primary})


def _forward_return(
    frame: pd.DataFrame,
    dividends: pd.DataFrame,
    horizon: int,
) -> tuple[pd.Series, pd.Series, np.ndarray, pd.Series]:
    """Eq. (7) forward risk-adjusted return for one horizon.

    Returns ``(R, price_fwd, dividends_fwd, rf_cum)`` with
    ``R = (P_{t+H} + sum d - P_t) / P_t - rf_cum``. Strictly forward-looking
    because it is a *label* (never a feature).
    """
    close = frame["close"].astype(float)
    price_fwd = close.shift(-horizon)

    div = np.zeros(len(frame))
    if len(dividends):
        tick = str(frame["ticker"].iloc[0]).upper()
        t_div = dividends[dividends["ticker"] == tick]
        dates = frame["date"].to_numpy()
        for _, row in t_div.iterrows():
            in_window = (dates > np.datetime64(row["ex_date"])) & (
                dates <= np.datetime64(row["ex_date"] + np.timedelta64(horizon, "D"))
            )
            div[in_window] += float(row["dividend"])

    rf_cum = frame["rf_daily"].rolling(horizon, min_periods=horizon).sum().shift(-horizon)
    ret = (price_fwd + div - close) / close - rf_cum
    return ret, price_fwd, div, rf_cum


def compute_target(
    features: pd.DataFrame,
    rf: pd.DataFrame,
    dividends: pd.DataFrame,
    cfg: dict,
) -> pd.DataFrame:
    """Attach the Eq. (7) forward returns for every horizon in ``target_horizons``.

    The primary horizon (``target.horizon``, default 30) also keeps the canonical
    ``R30`` name plus the ``price_fwd`` / ``dividends_fwd`` / ``rf_cum`` columns
    used by earlier experiments; each horizon additionally gets a generic
    ``R{h}`` column so the multi-horizon and period-mark evaluation can share one
    panel. The bounded labels are attached separately by
    :func:`standardize_target`, which needs the pooled panel for training-window
    statistics.
    """
    target_cfg = cfg.get("target", {})
    primary = int(target_cfg.get("horizon", 30))

    frame = features.merge(rf[["date", "rf_daily"]], on="date", how="left")
    for horizon in target_horizons(cfg):
        ret, price_fwd, div, rf_cum = _forward_return(frame, dividends, horizon)
        frame[f"R{horizon}"] = ret
        if horizon == primary:
            frame["price_fwd"] = price_fwd
            frame["dividends_fwd"] = div
            frame["rf_cum"] = rf_cum
    return frame



def train_window_zstats(panel: pd.DataFrame, cfg: dict, col: str = "R30") -> tuple[float, float, int]:
    """``(mean, std, n)`` of ``col`` over the training window only.

    Eq. (8) standardises the forward return over ``T_train``. Using full-sample
    statistics instead would let test-period information enter the training
    labels, so the reference moments are restricted here. ``col`` defaults to the
    canonical ``R30`` but accepts any ``R{h}`` for the multi-horizon labels.
    """
    exp = cfg.get("experiment", {})
    rows = panel["date"].between(exp["train_start"], exp["train_end"])
    reference = panel.loc[rows, col].to_numpy(dtype=float)
    reference = reference[np.isfinite(reference)]
    if reference.size == 0:
        raise ValueError(
            "no finite R30 values inside the training window -- check experiment.train_start/train_end"
        )
    std = float(reference.std(ddof=0))
    return float(reference.mean()), std, int(reference.size)


def _horizon_label(panel: pd.DataFrame, col: str, cfg: dict, values: pd.Series) -> pd.Series:
    """Bounded / rank-normalised label for one horizon (Eq. (10) or rank mode).

    ``target.label_mode = 'tanh'`` (default) reproduces Eq. (10)
    ``Y = 50 [1 + tanh(scale/50 * Z)]`` with ``Z`` scaled by *training-window*
    moments of the (optionally peer-demeaned) target. ``'rank'`` maps each date's
    cross-section to its percentile in ``[0, 100]`` -- a cross-sectional ranking
    target aligned with the Rank-IC objective -- falling back to the
    training-window empirical percentile on dates with fewer than two names.
    """
    target_cfg = cfg.get("target", {})
    mode = str(target_cfg.get("label_mode", "tanh")).lower()
    scale = float(target_cfg.get("tanh_scale", 15.0))

    if bool(target_cfg.get("peer_relative", False)):
        values = values - panel.groupby("date")[col].transform("mean")

    exp = cfg.get("experiment", {})
    train_rows = panel["date"].between(exp["train_start"], exp["train_end"]).to_numpy()
    raw = values.to_numpy(dtype=float)

    if mode == "rank":
        frame = panel.assign(_v=raw)
        counts = frame.groupby("date")["_v"].transform(lambda s: s.notna().sum())
        cross = (frame.groupby("date")["_v"].rank(pct=True) * 100.0).to_numpy()
        ref = np.sort(raw[train_rows & np.isfinite(raw)])
        if ref.size:
            fallback = np.searchsorted(ref, raw, side="right") / ref.size * 100.0
        else:
            fallback = np.full(len(raw), 50.0)
        return pd.Series(np.where(counts.to_numpy() >= 2, cross, fallback), index=panel.index)

    if mode != "tanh":
        raise ValueError("target.label_mode must be 'tanh' or 'rank'")

    ref = raw[train_rows & np.isfinite(raw)]
    mean = float(ref.mean()) if ref.size else 0.0
    std = float(ref.std(ddof=0)) if ref.size else 1.0
    z = (values - mean) / (std if std > 0 else 1.0)
    return 50.0 * (1.0 + np.tanh(scale / 50.0 * z))


def standardize_target(panel: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Attach the bounded label of Eq. (10) plus one label per horizon.

    The canonical ``Z30`` / ``Y_target`` use the primary horizon (Eq. (10)); every
    ``h`` in :func:`target_horizons` additionally gets a ``Y_target_{h}`` column
    (and, for the primary, the same ``tanh`` transform reproduces ``Y_target``
    exactly). Labels are mapped through *training-window* moments only, so no test
    information reaches them.
    """
    primary = int(cfg.get("target", {}).get("horizon", 30))
    pcol = "R30" if "R30" in panel.columns else f"R{primary}"
    if pcol not in panel.columns:
        raise KeyError("panel must carry an 'R30' (or 'R<horizon>') column")

    out = panel.copy()
    mean, std, _ = train_window_zstats(out, cfg, col=pcol)
    out["Z30"] = (out[pcol] - mean) / (std if std > 0 else 1.0)
    out["Y_target"] = _horizon_label(out, pcol, cfg, out[pcol])
    for horizon in target_horizons(cfg):
        col = f"R{horizon}"
        if col in out.columns:
            out[f"Y_target_{horizon}"] = _horizon_label(out, col, cfg, out[col])
    return out


def winsor_limits(panel: pd.DataFrame, cfg: dict) -> dict[str, tuple[float, float]]:
    """Train-window percentile bounds of the unbounded feature columns.

    ``order_flow`` (CLV/tick) and ``clv_5`` already live in ``[-1, 1]`` and are
    left untouched. The bounds are estimated on the training window alone and then
    applied to every split, so no test information leaks into the model while
    test-period extremes still cannot drive the consequents outside the score
    domain. With the default (three-input) configuration this covers exactly
    ``dev_price`` / ``dev_volume``.
    """
    features_cfg = cfg.get("features", {})
    low_pct = float(features_cfg.get("winsor_low_pct", 1.0))
    high_pct = float(features_cfg.get("winsor_high_pct", 99.0))
    exp = cfg.get("experiment", {})
    rows = panel[panel["date"].between(exp["train_start"], exp["train_end"])]
    limits = {}
    for column in _winsor_columns(cfg):
        if column not in rows.columns:
            continue
        values = rows[column].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        limits[column] = (
            float(np.percentile(values, low_pct)),
            float(np.percentile(values, high_pct)),
        )
    return limits


def apply_winsor(panel: pd.DataFrame, limits: dict[str, tuple[float, float]]) -> pd.DataFrame:
    """Clip ``dev_price``/``dev_volume`` to precomputed percentile bounds."""
    out = panel.copy()
    for column, (low, high) in limits.items():
        out[column] = out[column].clip(low, high)
    return out


# ---------------------------------------------------------------------- panel
# ------------------------------------------------------------------ feature set
# The three canonical Section-3.1 inputs (always on).
_BASE_FEATURE_COLUMNS = ("dev_price", "dev_volume", "order_flow")
# Tier-2 optional multi-scale microstructure inputs (OHLC-derivable, all trailing).
_EXTRA_FEATURE_CATALOG = ("dev_price_60", "ret_5", "vol_20", "clv_5", "amihud")
# Tier-2 optional slow/fundamental inputs attached as-of (point-in-time) below.
FUNDAMENTAL_COLUMNS = ("ebitda_growth", "nd_ebitda", "fcfe_to_ebitda")
# Columns whose scale is already bounded and which must NOT be winsorised.
_BOUNDED_COLUMNS = {"order_flow", "clv_5"}

# Backwards-compatible default: modules that ``from src.data_loader import
# FEATURE_COLUMNS`` keep receiving the three canonical inputs. Anything that must
# honour the Tier-2 enrichment calls :func:`feature_columns(cfg)` instead.
FEATURE_COLUMNS = _BASE_FEATURE_COLUMNS


def feature_columns(cfg: dict) -> tuple[str, ...]:
    """Feature vector for the TSK engine under the current configuration.

    Always starts from the three Section-3.1 inputs, then appends any configured
    Tier-2 enrichment: multi-scale microstructure (``features.extra_features``) and
    slow fundamentals (``features.fundamentals: true``).
    """
    feats = cfg.get("features", {})
    columns = list(_BASE_FEATURE_COLUMNS)
    for name in feats.get("extra_features") or []:
        if name not in _EXTRA_FEATURE_CATALOG:
            raise ValueError(
                f"unknown features.extra_features entry {name!r}; "
                f"known: {_EXTRA_FEATURE_CATALOG}"
            )
        if name not in columns:
            columns.append(name)
    if bool(feats.get("fundamentals", False)):
        columns.extend(c for c in FUNDAMENTAL_COLUMNS if c not in columns)
    return tuple(columns)


def _winsor_columns(cfg: dict) -> list[str]:
    """Feature columns to winsorise (everything unbounded in the active vector)."""
    return [c for c in feature_columns(cfg) if c not in _BOUNDED_COLUMNS]


def build_fundamentals_features(
    dates,
    ticker: str,
    financials: pd.DataFrame | None,
    cfg: dict,
) -> pd.DataFrame:
    """Point-in-time (as-of) slow fundamentals for one ticker -- no lookahead.

    Each disclosure becomes effective ``features.fundamentals_lag_days`` trading
    days after its period end (a conservative reporting lag), so a calendar day
    never sees figures that were not yet public. The ratio block
    (``nd_ebitda``, ``fcfe_to_ebitda``) is carried from every disclosure;
    ``ebitda_growth`` is period-on-period growth measured on the *annual* totals
    only, so it is a clean year-on-year figure carried forward. Returns a frame
    keyed on ``date`` with :data:`FUNDAMENTAL_COLUMNS`.
    """
    lag_days = int(cfg.get("features", {}).get("fundamentals_lag_days", 90))
    frame = pd.DataFrame({"date": pd.to_datetime(pd.Series(dates))}).reset_index(drop=True)
    frame = frame.sort_values("date").reset_index(drop=True)
    nan_col = pd.Series(np.full(len(frame), np.nan), name="date")

    def _empty() -> pd.DataFrame:
        out = frame[["date"]].copy()
        for column in FUNDAMENTAL_COLUMNS:
            out[column] = np.nan
        return out

    if financials is None or not len(financials):
        return _empty()
    rows = financials[financials["ticker"] == str(ticker).upper()].copy()
    if not len(rows):
        return _empty()

    rows["effective"] = rows["period_end"] + pd.Timedelta(days=lag_days)
    rows = rows.sort_values(["effective", "is_annual"]).drop_duplicates("effective", keep="last")
    rows["nd_ebitda"] = rows["net_debt"] / rows["ebitda"].replace(0.0, np.nan)
    rows["fcfe_to_ebitda"] = rows["fcfe"] / rows["ebitda"].replace(0.0, np.nan)

    merged = pd.merge_asof(
        frame, rows[["effective", "nd_ebitda", "fcfe_to_ebitda"]],
        left_on="date", right_on="effective", direction="backward",
    )

    annual = rows[rows["is_annual"]].sort_values("effective").copy()
    if len(annual) >= 2:
        annual["ebitda_growth"] = annual["ebitda"].pct_change().replace(
            [np.inf, -np.inf], np.nan
        )
        growth = pd.merge_asof(
            frame, annual[["effective", "ebitda_growth"]],
            left_on="date", right_on="effective", direction="backward",
        )[["ebitda_growth"]]
    else:
        growth = nan_col.to_frame("ebitda_growth")
    merged = pd.concat([merged[["date", "nd_ebitda", "fcfe_to_ebitda"]], growth], axis=1)
    return merged[["date", *FUNDAMENTAL_COLUMNS]].replace([np.inf, -np.inf], np.nan)


def _fill_fundamentals(panel: pd.DataFrame) -> pd.DataFrame:
    """Impute slow-fundamental gaps so no rows are lost.

    ``ebitda_growth`` in particular is undefined until two annual disclosures are
    visible, and every ratio is undefined before the first disclosure becomes
    effective. Gaps are filled by the cross-sectional (per-date) median, then the
    overall median, then ``0.0`` -- never with future information.
    """
    out = panel.copy()
    for column in FUNDAMENTAL_COLUMNS:
        if column not in out.columns:
            continue
        out[column] = out[column].replace([np.inf, -np.inf], np.nan)
        out[column] = out[column].fillna(out.groupby("date")[column].transform("median"))
        overall = out[column].median()
        out[column] = out[column].fillna(0.0 if pd.isna(overall) else overall)
    return out


def build_panel(cfg: dict, split: str = "all") -> pd.DataFrame:
    """Pooled long panel ``[date, ticker, features..., R30, Z30, Y_target, pre_agm]``.

    ``split`` selects ``train`` / ``test`` / ``all`` per the ``experiment``
    windows; ``pre_agm`` arms the Section-3.4 masking indicator from
    ``experiment.agm_dates`` and ``experiment.pre_agm_days``.
    """
    ohlc = load_all_ohlc(cfg)
    rf = risk_free_daily(load_tonia(cfg))
    dividends = load_dividends(cfg)
    with_fund = bool(cfg.get("features", {}).get("fundamentals", False))
    financials = load_financials(cfg) if with_fund else None

    pre_days = int(cfg.get("experiment", {}).get("pre_agm_days", 45))
    agm_dates = cfg.get("experiment", {}).get("agm_dates", {})

    parts = []
    for ticker, bars in ohlc.items():
        feats = compute_target(compute_features(bars, cfg), rf, dividends, cfg)
        if with_fund:
            fund = build_fundamentals_features(feats["date"], ticker, financials, cfg)
            feats = feats.merge(fund, on="date", how="left")
        spec = agm_dates.get(ticker, [])
        stamps = spec if isinstance(spec, (list, tuple)) else [spec]
        mask = np.zeros(len(feats), dtype=bool)
        dates = feats["date"]
        for stamp in stamps:
            agm_date = pd.Timestamp(stamp)
            if pd.isna(agm_date):
                continue
            mask |= (
                (dates >= agm_date - pd.Timedelta(days=pre_days)) & (dates < agm_date)
            ).to_numpy()
        feats["pre_agm"] = mask.astype(float)
        parts.append(feats)

    panel = pd.concat(parts, ignore_index=True)
    drop_columns = [c for c in feature_columns(cfg) if c not in FUNDAMENTAL_COLUMNS]
    panel = panel.dropna(subset=drop_columns)
    if with_fund:
        panel = _fill_fundamentals(panel)
    # Winsorize dev_price/dev_volume to train-window percentiles: fit on the
    # training window only, applied to every split (no test leakage), which also
    # keeps Proposition 1's constant C = max|y_k - Y_raw| inside the score domain.
    if bool(cfg.get("features", {}).get("winsorize", True)):
        panel = apply_winsor(panel, winsor_limits(panel, cfg))
    # Eq. (10) labels use training-window moments only (no test leakage).
    panel = standardize_target(panel, cfg)
    exp = cfg.get("experiment", {})
    if split == "train":
        panel = panel[panel["date"].between(exp["train_start"], exp["train_end"])]
    elif split == "test":
        panel = panel[panel["date"].between(exp["test_start"], exp["test_end"])]
    return panel.reset_index(drop=True)
