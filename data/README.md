# Data Requirements — Adaptive TSK Equity Scoring (full article completion)

Project root: `c:\adaptive-tsk-equity-scoring`
Paper (authoritative spec): `paper/sn-article-template\name.tex`

> The engine is first-order TSK fuzzy inference over three OHLCV features;
> every claim below that depends on a data source is tied to the paper section it gates.

## 1. Core inputs (reproduce the paper's headline claims)

| # | Source / file | What it contains | Status | Gates |
|---|---|---|---|---|
| 1 | `data/raw/ohlc/{AIRA,KCEM,KEGC,KMGZ,KZAP,KZTK,KZTO}.csv` | L1 daily bars (Date, Price, Open, High, Low, Vol., Change %) from Investing.com | **Present** — 2022-01-05→2025-12-31 (AIRA from 2024-02-12, KMGZ from 2022-12-09). Covered by `src/data_loader.py` (`load_ohlc`). | Engine, all ICs, portfolio, §4.4 masking evaluation |
| 2 | `data/raw/financials/financials-{2022..2025}.csv` | EBITDA / Net Debt / FCFE (mln KZT) per year; Q1–Q4 and H1/H2 legs + annual "За <year> год" totals | **Present** — 101 tidy rows; delimiter-agnostic (tab / semicolon / comma). Covered by `load_financials`. | `Phi(ND/EBITDA, FCFE)`, payout table, fundamentals-only benchmark |
| 3 | `data/raw/macro/tonia_rbk.xlsx` | Daily TONIA close (RBK export; header row embedded; `Дата`/`Закрытие` headers) | **Present** — consumed by `load_tonia` + `risk_free_daily` (`rf_daily = TONIA/100/252`) | R30 risk-adjusted label (Eq. 7) |
| 4 | `data/raw/macro/rate_nbk.xlsx` | Daily USD/KZT reference rate (NBK; contextual only, not an engine input) | **Present** — `load_fx` | §4.3 Sharpe context only |
| 5 | `data/raw/financials/dividends.csv` | **Present** — 33 collected rows (`ticker, ex_dividend_date, dividend_per_share_kzt, declaration_date, status`); `load_dividends` accepts this schema plus the canonical `ticker, ex_date, dividend`, excludes `cancelled` declarations, coerces `N/A` → dropped. 26 paid rows load (AIRA 1, KEGC 8, KMGZ 4, KZAP 4, KZTK 5, KZTO 4; KCEL 0 — its 2022 107.50 row is `cancelled`) | Consumer: `load_dividends` → dividend-adjusted R30 labels (Eq. 7) and dividend-adjusted portfolio returns (`dividends_file_present: true`). Flag: KZTK 2025-05-25 26,664.88 KZT/share verified real (~78% one-day yield; price 58,400 → 39,999.99 → 34,279) — dominates 2025 KZTK labels; dropping KZTK cuts top-3 Sharpe to 1.40 | §4.1 label quality; §4.3/§4.4/§5 portfolio figures |

## 2. Corporate decision input (Section 4.1 / 4.2)

| # | Source / file | What it contains | Status | Gates |
|---|---|---|---|---|
| 6 | `data/raw/financials/actual_decisions.csv` | **Present** — 28 collected rows FY2021–2024 (`ticker, fiscal_year, decision_date, approved_pct, source_url, band_lo, band_hi, policy_url`); `load_board_decisions(cfg, fy)` parses `%`/Cyrillic/`N/A` sentinels to 0–1 fractions. FY2024 approved: AIRA 75%, KCEL 0%, KEGC 76%, KMGZ 27%, KZAP 75%, KZTK 100%, KZTO 80% | Consumer: `experiments/payout_alignment.py::_csv_actuals` (CSV-first, paper-transcribed `ACTUAL` dict as fallback). Denominator caveat: approved = % of profit, model target = % of FCFE — KZTK 100%-of-profit vs −44,500 FCFE is a units mismatch, not a missed call | Analysis 7 payout-alignment readout; Table 7 |

Notes on (6): the 2025 AGM dates are documented in `config.yaml` under `experiment.agm_dates`
(treated as month/day approximations; exact KASE disclosure dates should replace them when
available). The collected file sits under the broker-exports licensing agreement and is
**not** redistributable to the public repository; the public repo keeps code + derived outputs only.

## 3. Optional / extended microstructure (not required for the core article)

| # | Source / file | What it contains | Status | Gates |
|---|---|---|---|---|
| 7 | Level-2 orderbook / depth (`{ticker}.depth.csv` with bid/ask sizes) | **Not required.** The repo is L1-only: `src/data_loader.py` has no depth code and `config.yaml` `features.extra_features: []`. Needed only if the paper gains order-book imbalance / mid-price depth / L2-vs-L1 liquidity arguments | Optional | Extended §3/§4 microstructure content |
| 8 | KZTO retained-earnings bridge (retained earnings / cash balance time series 2022–2024) | Not in repo; sources: KZTO annual report / KASE annual financial statement, KZTO investor presentation | Optional | §4.2 qualitative explanation of the KZTO mismatch row only |
| 9 | KASE market-wide data (index level, total volume) | Not used by any script | Optional | Market context only |

## 4. Reproducibility

- The public repository contains: `src/`, `experiments/`, `data/raw/` (items 1–4 plus the two
collected files), `data/processed/` (panel features + derived labels), `outputs/` (models, logs, csv_exports), and `paper/sn-article-template/`.
- Both previously missing files (5, 6) are now present: (5) dividend-adjusted R30 labels flow
through `load_dividends`; (6) CSV-approved FY2024 singles anchor Analysis 7 via `_csv_actuals`,
with the paper-transcribed ranges as fallback.
