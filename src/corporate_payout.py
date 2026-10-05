"""Corporate payout mapping (Section 4.1, Eqs. (12)-(13) of the paper).

The daily attractiveness score is mapped to a payout-ratio target via

    PR_target = psi(Y_attr) * Phi(Net Debt / EBITDA, FCFE)

with ``psi(Y) = clamp(Y / 100, 0, 1)`` and the leverage / FCFE penalty

    Phi(L, F) = 1 / (1 + exp(kappa (L - tau)))   if F > 0, else 0

using ``kappa = 3.0`` and ``tau = 1.0`` (config: ``payout``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["psi", "phi", "payout_table", "latest_fundamentals"]


def psi(y: np.ndarray) -> np.ndarray:
    """Score-to-payout squashing ``psi(Y) = clamp(Y/100, 0, 1)``."""
    return np.clip(np.asarray(y, dtype=float) / 100.0, 0.0, 1.0)


def phi(leverage: np.ndarray, fcfe: np.ndarray, kappa: float = 3.0, tau: float = 1.0) -> np.ndarray:
    """Leverage / FCFE penalty of Eq. (13)."""
    leverage = np.asarray(leverage, dtype=float)
    fcfe = np.asarray(fcfe, dtype=float)
    logistic = 1.0 / (1.0 + np.exp(kappa * (leverage - tau)))
    return np.where(fcfe > 0, logistic, 0.0)


def latest_fundamentals(financials: pd.DataFrame) -> pd.DataFrame:
    """Most recent reported fundamentals per ticker (full-year when published).

    ``load_financials`` sorts by ``(ticker, period_end, is_annual)``, so the last
    row per ticker is the latest disclosure and, on the same period end, the
    annual ``За <year> год`` total rather than the Q4/H2 leg.
    """
    return financials.groupby("ticker").tail(1).set_index("ticker")


def payout_table(
    financials: pd.DataFrame,
    scores: pd.DataFrame,
    kappa: float = 3.0,
    tau: float = 1.0,
) -> pd.DataFrame:
    """Payout recommendations per ticker.

    ``scores``: frame indexed by ticker with a ``Y_attr`` column (e.g. the mean
    or latest out-of-sample score). Returns ``FCFE, ND_EBITDA, PR_target``.
    """
    fund = latest_fundamentals(financials)
    joined = fund.join(scores[["Y_attr"]], how="inner").reset_index()
    leverage = joined["net_debt"] / joined["ebitda"].replace(0.0, np.nan)
    joined["ND_EBITDA"] = leverage
    joined["PR_target"] = psi(joined["Y_attr"].to_numpy()) * phi(
        leverage.to_numpy(), joined["fcfe"].to_numpy(), kappa, tau
    )
    return joined[
        ["ticker", "fcfe", "ND_EBITDA", "Y_attr", "PR_target"]
    ].sort_values("ticker").reset_index(drop=True)
