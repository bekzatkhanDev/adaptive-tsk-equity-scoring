# Adaptive TSK Equity Scoring

Implementation of **"An Adaptive TSK Fuzzy-Inference Scoring Framework with Intraday
Range-Imbalance Masking for Emerging Equity Markets"** (see
`paper/sn-article-template/name.tex` — the camera-ready article is the authoritative
specification of every equation referenced below).

A first-order Takagi–Sugeno–Kang fuzzy scorer maps three daily OHLCV features to an
equity-attractiveness score `Y_attr ∈ [0, 100]`, trained with a hybrid ANFIS
procedure and protected during Pre-AGM windows by stochastic rule-weight masking.

## Pipeline

```
data/raw/ohlc/{TICKER}.csv                     Investing.com daily bars (Date, Price, Open, High, Low, Vol., Change %)
data/raw/financials/financials-{year}.csv     EBITDA / Net Debt / FCFE per reporting period (mln KZT), one file per year
data/raw/macro/tonia_rbk.xlsx                 daily TONIA close (RBK export; header row inside the sheet)
data/raw/macro/rate_nbk.xlsx                  daily USD/KZT reference rate (NBK; contextual series)
        │  src/data_loader.py      -- parse, align, feature build (Eqs. 1-4)
        ▼
panel features: Dev_Price, Dev_Volume, CLV  (FEATURE_COLUMNS)
        │  src/TSK_engine.py       -- K=5 subtractive clustering, Gaussian MFs (Eqs. 5-6)
        │  src/anfis_trainer.py    -- hybrid: LS consequents (Eq. 9) + width GD (Eq. 11)
        │  src/masking.py          -- Pre-AGM truncated-Gaussian rule masking (Eqs. 14-16)
        ▼
outputs/models/*.json   outputs/logs/*   outputs/csv_exports/*
        │  src/backtest.py         -- Rank ICs, block bootstrap, top-3 net Sharpe
        │  src/corporate_payout.py -- PR = psi(Y) * Phi(ND/EBITDA, FCFE)  (Eqs. 12-13)
        ▼
paper/sn-article-template/kmgz_2025_out_of_sample.csv  (Fig. 2 dataset: day, Y_attr, R30_scaled)
```

## Quick start

```powershell
python -m pip install -r requirements.txt

# 1. pooled-panel training on the train window (2024)
python experiments/train_anfis_panel.py

# 2. out-of-sample evaluation (Rank ICs, bootstrap CI, cost-tier Sharpe, OLS baseline)
python experiments/evaluate_2025_oos.py --ticker KMGZ --publish-paper-csv

# 3. adversarial reconstruction (SVR: unmasked vs masked vs shuffled control)
python experiments/SVR_adversarial_reconstruction.py --alpha 0.05

# 4. payout recommendations vs. actual board decisions (paper Table 1)
python experiments/payout_recommendations.py

# 5. per-ticker processed features (Eqs. 1-10) + multi-horizon labels -> data/processed/
python experiments/build_features.py

# 6. multi-horizon fit + weekly/monthly/quarterly/annual period marks
python experiments/evaluate_multihorizon.py

# 7. chronological hyper-parameter scan (2024-H1 fit / 2024-H2 validation)
python experiments/tune_tsk.py

# 8. Tier-2 feature-set comparison (baseline vs microstructure vs fundamentals)
python experiments/compare_feature_sets.py

# single-ticker variant
python experiments/train_anfis_single.py --ticker KMGZ
```

## Tier-1 improvements (multi-horizon labels & period marks)

The scorer is evaluated at several horizons and at board-relevant cadences rather
than only daily:

* `target.horizons` (default `[5, 21, 63, 252]`) adds `R{h}` / `Y_target_{h}`
  columns on top of the canonical `R30` / `Y_target` (the primary `target.horizon`
  is always included).
* `target.label_mode` selects the Eq. (10) `tanh` label or a cross-sectional
  `rank` (percentile in `[0, 100]`); `target.peer_relative` demeans `R_h` within
  each date first.
* `src.backtest.period_marks` / `mark_rank_ic` compute the weekly / monthly /
  quarterly / annual **mark** (aggregate of daily scores over a period) and its
  cross-sectional Rank IC against the *next* period's forward return -- the mark a
  corporate treasury board would act on. `evaluate_multihorizon.py` reports both
  the per-horizon fit and the mark table.

Observed on the current 2024-train / 2025-test sample: per-horizon pooled Rank IC
is flat to slightly negative, but the **monthly mark IC is +0.28 (SE 0.12) and the
quarterly mark IC is +0.33 (SE 0.15)** -- i.e. the signal is far clearer at
rebalance cadence than at the daily horizon, which is the intent of the mark
statistics.

## Tier-2 improvements (feature enrichment)

The input vector can be enriched without touching the canonical model: both
switches default to **off**, so `feature_columns(cfg)` returns the three
Section-3.1 inputs and every existing artifact is unchanged.

* `features.extra_features` — multi-scale trailing microstructure, any of
  `dev_price_60`, `ret_5`, `vol_20`, `clv_5`, `amihud`.
* `features.fundamentals: true` — as-of (point-in-time) slow fundamentals
  (`ebitda_growth`, `nd_ebitda`, `fcfe_to_ebitda`) attached with a
  `fundamentals_lag_days` (default 90) reporting lag, so no calendar day sees a
  disclosure that was not yet public.
* `tsk.input_scaling: minmax` — train-fitted `[0, 1]` input scaling; required when
  heterogeneous Tier-2 columns share the vector. `raw` (default) is the identity
  map, leaving the canonical model bit-for-bit unchanged.

`experiments/compare_feature_sets.py` trains the same protocol on each vector:

| Set | Inputs | Free params | Pooled Rank IC | Net Sharpe (15 bps) |
|---|---|---|---|---|
| baseline (canonical) | 3 | 35 | −0.013 | 0.100 |
| **microstructure** | 8 | 85 | **+0.186** | **+1.646** |
| fundamentals | 6 | 65 | −0.013 | −0.368 |
| full | 11 | 115 | +0.032 | +0.877 |

The **multi-scale microstructure set is the one real out-of-sample gain found so
far**: pooled Rank IC rises from ≈0 to +0.186 and the base-tier net Sharpe from
0.10 to 1.65, at the cost of more parameters (35 → 85) and a looser fit to the
Eq. (10) target (RMSE 18.2 → 21.4, the fit-vs-rank trade-off the article already
discusses). Slow fundamentals alone add nothing, and diluting the vector with
them (`full`) *reduces* the microstructure gain. The period-mark ICs do **not**
improve in step (microstructure monthly +0.19 vs baseline +0.28), so the gain is
horizon-dependent and rests on a single out-of-sample year.

## Current results (2024 train → full-year 2025 out-of-sample)

The raw bars now cover **2024-01-03 → 2025-12-31**, so the paper's canonical **2025
out-of-sample test window is fully populated** and the reported numbers are real results
of the pipeline, not plumbing checks. Training uses the 2024 calendar year (1,575 pooled
asset-days); testing uses all of 2025 (1,722 asset-days over 246 trading days, 1,512 with
an observable 30-day forward label).

| Metric | TSK (CLV) | OLS baseline |
|---|---|---|
| Total (free) parameters | 50 (35) | 4 (4) |
| RMSE / MAE vs. bounded target | 15.17 / 11.89 | 14.99 / 11.78 |
| Per-asset daily Rank IC (mean ± SE) | 0.136 ± 0.085 | — |
| Pooled cross-sectional monthly Rank IC | 0.188 ± 0.091 | — |
| Pooled asset-day Rank IC [95% CI] | 0.047 [−0.055, 0.164] | −0.011 |
| Net Sharpe, 15 / 25 / 35 / 50 bps | 1.23 / 1.18 / 1.14 / 1.07 | 0.20 / 0.16 / 0.12 / 0.05 |
| SVR reconstruction R² (Pre-AGM, α=0 → 0.05) | 0.268 → −0.027 | — |

Caveats: only eleven monthly rebalances exist, so the base-tier Sharpe bootstrap interval
is wide ([−0.84, 3.59]); monthly re-calibration on a 504-day window is not exercised
because 2022–2023 bars are still absent.

To move to the canonical training window: add 2022–2023 CSVs (same Investing.com format)
to `data/raw/ohlc/` and set in `config.yaml`:

```yaml
train_start: "2022-01-01"
train_end:   "2024-12-31"
```

No code changes are needed. The AGM dates in `experiment.agm_dates` (lists are supported)
should likewise be replaced with the exact KASE disclosure dates.

## Verified numerics

* Premise gradients (`log_sigma` and `centers`) match central finite differences to
  **~1e-8 relative error** (log-width parameterisation keeps widths positive and stable).
* Masking bias matches Proposition 1's bound: worst case `100 * (0.025)^2 = 0.0625`; the
  α = 0 SVR run reproduces the unmasked R² exactly (0.268), confirming the masking path is
  a no-op when disarmed.
* Financials parsing is delimiter-agnostic (2024 semicolon / 2025 comma), handles annual
  `За <year> год` totals alongside Q1–Q4 and H1/H2 legs, and yields 47 tidy rows.

## Known environment limitations

* **No LaTeX toolchain installed** — compile the article on Overleaf or any TeX Live
  install; `paper/sn-article-template/` holds the official Springer Nature class files.
* Financial statements are quarterly/half-year (Q1–Q4, H1/H2) and are mapped to the
  latest-available reporting period per date (no lookahead); a dividends CSV is optional
  (`paths.raw_dividends`, currently zeros).
