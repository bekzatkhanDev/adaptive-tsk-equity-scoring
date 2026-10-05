"""Adaptive TSK equity scoring -- article-aligned implementation.

Modules
-------
``data_loader``       raw ingestion (Investing.com / RBK xlsx) + Section-3.1 features
``TSK_engine``        first-order TSK inference, subtractive clustering, Eqs. (4)-(6)
``anfis_trainer``     ANFIS hybrid training (LS consequents + width gradients)
``masking``           Pre-AGM stochastic rule-weight masking (Section 3.4)
``corporate_payout``  payout mapping psi * Phi (Eqs. (12)-(13))
``backtest``          Rank IC, block bootstrap, cost-tier net Sharpe (Section 4)
"""

__all__ = ["data_loader", "TSK_engine", "anfis_trainer", "masking",
           "corporate_payout", "backtest"]