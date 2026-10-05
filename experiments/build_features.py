"""Export per-ticker processed features (Eqs. (1)-(10) of the paper).

Writes ``data/processed/<TICKER>_features.csv`` for each study equity plus a
pooled ``data/processed/panel_features.csv``, so the Section-3.1 feature
construction and the multi-horizon labels can be inspected without running a
model. Uses the canonical :func:`build_panel` pipeline, so the exported panel is
exactly what the trainer and evaluator consume (winsorised features, Eq. (10)
labels, Pre-AGM flags, and one ``R{h}`` / ``Y_target_{h}`` block per horizon).

Example
-------
python experiments/build_features.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import build_panel, feature_columns, load_config, target_horizons

# Config-aware active input vector so the export carries whatever the model trains on.
FEATURE_COLUMNS = feature_columns(load_config())

# Columns carried by ``build_panel`` that we expose (the label block is extended
# with the multi-horizon ``R{h}`` / ``Y_target_{h}`` columns in ``main``).
BASE_COLUMNS = [
    "date", "ticker", "close", "volume",
    *FEATURE_COLUMNS, "pre_agm", "R30", "Z30", "Y_target",
]


def main() -> None:
    cfg = load_config()
    out_dir = Path(cfg["paths"]["processed"])
    out_dir.mkdir(parents=True, exist_ok=True)

    panel = build_panel(cfg, split="all")
    horizons = target_horizons(cfg)
    wanted = BASE_COLUMNS + [
        col for h in horizons for col in (f"R{h}", f"Y_target_{h}")
    ]
    columns: list[str] = []
    for col in wanted:
        if col in panel.columns and col not in columns:
            columns.append(col)

    for ticker, frame in panel.groupby("ticker"):
        export = frame[columns].sort_values("date")
        export.to_csv(out_dir / f"{ticker}_features.csv", index=False)
        print(
            f"{ticker}: {len(export)} rows, {int(export['Y_target'].notna().sum())} labelled, "
            f"{int(export['pre_agm'].sum())} pre-AGM -> {ticker}_features.csv"
        )

    pooled = panel[columns].sort_values(["date", "ticker"])
    pooled.to_csv(out_dir / "panel_features.csv", index=False)
    print(f"panel: {len(pooled)} rows -> panel_features.csv")


if __name__ == "__main__":
    main()
