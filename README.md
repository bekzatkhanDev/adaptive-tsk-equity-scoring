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

# 7. chronological hyper-parameter scan (train-window midpoint split:
#    2022-01-01..2023-07-01 fit / 2023-07-01..2024-12-31 validation)
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

`experiments/compare_feature_sets.py` trains the same protocol on each vector.
Single seed (the published default run, plain training panel):

| Set | Inputs | Free params | Pooled Rank IC | Net Sharpe (15 bps) |
|---|---|---|---|---|
| baseline (canonical) | 3 | 35 | −0.024 | +2.58 |
| **microstructure** | 8 | 85 | **+0.095** | +2.26 |
| fundamentals | 6 | 65 | −0.043 | +0.08 |
| full | 11 | 115 | +0.008 | +1.32 |

`--seeds N` re-trains every set on N i.i.d. row bootstraps of the 2022–2024 training
panel (seed = `project.seed` shifted by 0..N−1; the test panel is never
resampled). With N = 5 (`outputs/csv_exports/feature_set_comparison_5seed.csv`):

| Set | Pooled Rank IC (mean ± SE) | Seed spread | Net Sharpe 15 bps (mean ± SE) | Monthly mark IC (mean ± SE) |
|---|---|---|---|---|
| baseline | +0.019 ± 0.021 | [−0.047, +0.074] | +1.47 ± 0.25 | +0.19 ± 0.05 |
| **microstructure** | +0.045 ± 0.009 | [+0.031, +0.082] | **+2.17 ± 0.22** | +0.05 ± 0.02 |
| fundamentals | −0.041 ± 0.002 | [−0.047, −0.036] | −0.06 ± 0.08 | +0.11 ± 0.01 |
| full | **+0.049 ± 0.018** | [−0.010, +0.099] | +1.52 ± 0.25 | +0.10 ± 0.03 |

What survives the three-year retraining:

* **Microstructure enrichment is a real, if modest, ranking gain.** Its pooled Rank IC is
  positive on all five draws (+0.045 ± 0.009 vs the baseline's +0.019 ± 0.021), and on the
  bootstrap means it also carries the higher Sharpe (+2.17 ± 0.22 vs +1.47 ± 0.25). On the
  plain single-seed panel the ordering flips (+2.58 baseline vs +2.26 microstructure), so
  the Sharpe ordering is seed-dependent while the ranking ordering is not. The cost of the
  extra inputs is a looser fit to the Eq. (10) target (mean RMSE 12.29 → 15.13) and
  35 → 85 free parameters.
* **Slow fundamentals remain worthless here** — negative pooled Rank IC on every draw
  (−0.041 ± 0.002) and a net Sharpe indistinguishable from zero. With three years of
  disclosure history `ebitda_growth` is finally defined, yet the signal still does not
  appear.
* **`full` is no longer worse than `microstructure` on ranking** (it has the highest mean
  pooled IC, +0.049) but it gives back most of the Sharpe, so mixing fundamentals in buys
  no ranking and costs return.
* **Period marks still do not track the pooled IC**: baseline monthly mark IC
  (+0.19 ± 0.05) beats microstructure (+0.05 ± 0.02) even though microstructure wins on
  pooled IC. Ranking skill and rebalance-cadence skill are different things.

Caveats: the gains rest on one out-of-sample year (11 monthly rebalances); the
Deflated Sharpe check over all 20 (4 sets × 5 seeds) trials reports **DSR = 0.00**
(`SR_obs` 2.45 vs `SR*` 7.53), so the *portfolio-level* edge is not defensibly better
than chance once the number of configurations tried is accounted for. The ranking edge
is the defensible claim; the Sharpe edge is not.

## Current results (2022–2024 train → full-year 2025 out-of-sample)

The raw bars now cover **2022-01-05 → 2025-12-31** (AIRA only from 2024-02-12, KMGZ from
2022-12-09), so training uses the paper's canonical **three-year window**: 4,278 pooled
asset-days from 2022-02-07 → 2024-12-31 (start trimmed by the feature warm-up). Testing
uses all of 2025 — 1,722 asset-days over 246 trading days, 1,512 with an observable
30-day forward label.

| Metric | TSK (CLV) | OLS baseline |
|---|---|---|
| Total (free) parameters | 50 (35) | 4 (4) |
| RMSE / MAE vs. bounded target | 12.38 / 8.80 | 12.09 / 8.66 |
| Per-asset daily Rank IC (mean ± SE) | −0.045 ± 0.032 | — |
| Pooled cross-sectional monthly Rank IC | 0.438 ± 0.062 | — |
| Pooled asset-day Rank IC [95% CI] | −0.024 [−0.091, 0.078] | −0.046 |
| Net Sharpe, 15 / 25 / 35 / 50 bps | 2.58 / 2.54 / 2.50 / 2.44 | 0.84 / 0.79 / 0.74 / 0.67 |
| SVR reconstruction R² (Pre-AGM, α=0 → 0.05) | 0.284 → 0.282 | — |

Moving from one training year to three materially changes the picture: the model fits the
bounded target far better (RMSE 15.17 → 12.38) and the **monthly** rebalance signal becomes
strong (0.438 ± 0.062), while the *daily* pooled Rank IC remains statistically
indistinguishable from zero (−0.024 [−0.091, 0.078]) and still does not beat OLS as a
point forecast. The honest summary is therefore: **no daily forecasting skill, clear
monthly-cross-sectional ranking skill.**

Two caveats worth carrying into any write-up:

* Only eleven monthly rebalances exist, so the base-tier Sharpe bootstrap interval is wide
  ([0.67, 5.55]).
* The SVR reconstruction number (**0.284 → 0.282**) still shows *no* degradation under
  Pre-AGM masking — i.e. the article's §3.4/§4.4 headline (0.268 → −0.027) remains
  unreproducible and needs either a mechanism fix or an honest restatement.

Walk-forward validation (4 rolling 2025 folds, `outputs/logs/walk_forward_4fold.json`)
gives baseline DSR 0.82 / microstructure DSR 0.63, consistent with the single OOS run.

> **Paper sync status:** `paper/sn-article-template/name.tex` still carries the previous
> (2024-only) numbers and needs re-syncing to this run.

## Verified numerics

* Premise gradients (`log_sigma` and `centers`) match central finite differences to
  **~1e-8 relative error** (log-width parameterisation keeps widths positive and stable).
* Masking bias matches Proposition 1's bound: worst case `100 * (0.025)^2 = 0.0625`; the
  α = 0 SVR run reproduces the unmasked R² exactly (0.284), confirming the masking path is
  a no-op when disarmed.
* Financials parsing is delimiter-agnostic (2022/2023 tab, 2024 semicolon, 2025 comma),
  handles annual `За <year> год` totals alongside Q1–Q4 and H1/H2 legs, and yields 101
  tidy rows over 2022–2025.

## Known environment limitations

* **No LaTeX toolchain installed** — compile the article on Overleaf or any TeX Live
  install; `paper/sn-article-template/` holds the official Springer Nature class files.
* Financial statements are quarterly/half-year (Q1–Q4, H1/H2) and are mapped to the
  latest-available reporting period per date (no lookahead); a dividends CSV is optional
  (`paths.raw_dividends`, currently zeros).
