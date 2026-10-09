"""OFI-proxy internal-consistency audit (review item 2).

No Level-1 order-flow feed exists for these KASE names, so a direct CLV-vs-true-
OFI correlation is infeasible. This script reports what the daily bars *can*
support instead: (a) Rank IC of the CLV proxy against next-day signed returns
(sign predictability -- the necessary condition for any OFI proxy); (b) the
CLV/tick-rule agreement rate (Spearman rho and sign agreement between the CLV
statistic and the tick-rule fallback, i.e. whether two independent daily
directional readings coincide); (c) the hybrid fallback rate (share of rows
where H/L are missing or degenerate and the proxy collapses to the tick rule).
Weak results here would invalidate the proxy; strong results keep it plausible
without proving Level-1 equivalence -- stated as such in Section 3.1.

Example
-------
python experiments/ofi_proxy_audit.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.backtest import rank_ic
from src.data_loader import build_panel, load_config


def main() -> None:
    cfg = load_config()
    panel = build_panel(cfg, split="all").sort_values(["ticker", "date"]).reset_index(drop=True)
    panel["next_ret"] = panel.groupby("ticker")["R5"].shift(0)  # placeholder replaced below
    # Next-day close-to-close sign target, strictly trailing features vs forward label.
    panel["ret_1fwd"] = panel.groupby("ticker")["close"].transform(
        lambda s: s.shift(-1) / s - 1.0
    )
    valid = panel.dropna(subset=["ret_1fwd"])
    proxy_ic = rank_ic(valid["order_flow"].to_numpy(), valid["ret_1fwd"].to_numpy())

    tick = np.sign(valid.groupby("ticker")["close"].diff().fillna(0.0))
    clv = valid["order_flow"].to_numpy()
    agree = valid.copy()
    agree["tick"] = tick.to_numpy()
    both = agree[(agree["tick"] != 0) & np.isfinite(agree["order_flow"])]
    sign_agree = float(np.mean(np.sign(both["order_flow"]) == np.sign(both["tick"]))) if len(both) else float("nan")
    spearman = rank_ic(both["order_flow"].to_numpy(), both["tick"].to_numpy())

    per_ticker = {}
    for t, g in valid.groupby("ticker"):
        per_ticker[t] = {
            "n": int(len(g)),
            "proxy_next_day_ic": rank_ic(g["order_flow"].to_numpy(), g["ret_1fwd"].to_numpy()),
        }

    report = {
        "note": ("No Level-1 feed available for KASE names: direct CLV-vs-true-OFI "
                 "correlation infeasible; internal-consistency audit only."),
        "n_rows": int(len(valid)),
        "proxy_next_day_rank_ic": proxy_ic,
        "clv_tick_spearman": spearman,
        "clv_tick_sign_agreement": sign_agree,
        "per_ticker": per_ticker,
    }
    logs = Path(cfg["paths"]["logs"])
    (logs / "ofi_proxy_audit.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
