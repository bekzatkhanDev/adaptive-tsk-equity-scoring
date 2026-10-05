"""SVR adversarial reconstruction protocol (Section 4.4 of the paper).

An external HFT adversary fits an RBF Support Vector Regressor on observable
market inputs ``x_t`` to reconstruct the internal score. Reconstruction quality
(out-of-sample R^2) is compared between the unmasked scorer (``alpha = 0``)
and the Pre-AGM stochastic masking protocol (``alpha = 0.05``). A shuffled-
target control establishes the noise floor.

Example
-------
python experiments/SVR_adversarial_reconstruction.py --mask-rate 0.1 --alpha 0.05
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.svm import SVR

from src.data_loader import FEATURE_COLUMNS, build_panel, load_config
from src.masking import masked_firing_scores
from src.TSK_engine import TSKFuzzySystem


def reconstruct_r2(model: TSKFuzzySystem, x_train, y_train, x_test, y_test,
                   mask_rate: float, seed: int) -> float:
    """Adversary R^2: SVR (x -> score) trained on the training window."""
    svr = SVR(kernel="rbf", C=1.0, gamma="scale")
    if mask_rate > 0:
        rng = np.random.default_rng(seed)
        keep_train = rng.random(len(x_train)) >= mask_rate
        keep_test = rng.random(len(x_test)) >= mask_rate
        x_train, y_train = x_train[keep_train], y_train[keep_train]
        x_test, y_test = x_test[keep_test], y_test[keep_test]
        if len(x_train) < 10 or len(x_test) < 5:
            return float("nan")
    svr.fit(x_train, y_train)
    return float(r2_score(y_test, svr.predict(x_test)))


def run(cfg: dict, model: TSKFuzzySystem, alpha: float, mask_rate: float, seed: int) -> dict:
    """Reconstruction experiment restricted to the Pre-AGM windows.

    Section 4.4 protocol: the adversary only ever observes scores inside the
    Pre-AGM windows (``I_PreAGM = 1``), which is exactly where the masking is
    armed. It fits on the training-year Pre-AGM panel and is scored on the
    out-of-sample Pre-AGM panel.
    """
    train = build_panel(cfg, split="train")
    test = build_panel(cfg, split="test")
    train = train[train["pre_agm"] > 0]
    test = test[test["pre_agm"] > 0]
    if train.empty or test.empty:
        raise SystemExit("no Pre-AGM rows -- check experiment.agm_dates / pre_agm_days")

    x_train = train[list(FEATURE_COLUMNS)].to_numpy(dtype=float)
    x_test = test[list(FEATURE_COLUMNS)].to_numpy(dtype=float)

    y_train_clean = model.score(x_train)
    y_test_clean = model.score(x_test)
    rng = np.random.default_rng(seed)
    y_train_masked = masked_firing_scores(model, x_train, alpha=alpha, rng=rng)
    y_test_masked = masked_firing_scores(model, x_test, alpha=alpha, rng=rng)

    shuffle_rng = np.random.default_rng(seed + 1)
    y_test_shuffled = shuffle_rng.permutation(y_test_clean)

    return {
        "alpha": alpha,
        "mask_rate": mask_rate,
        "scope": "pre_agm_only",
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        "r2_unmasked": reconstruct_r2(model, x_train, y_train_clean, x_test, y_test_clean, mask_rate, seed),
        "r2_masked": reconstruct_r2(model, x_train, y_train_masked, x_test, y_test_masked, mask_rate, seed),
        "r2_shuffled_control": reconstruct_r2(model, x_train, y_train_clean, x_test, y_test_shuffled, mask_rate, seed),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--mask-rate", type=float, default=0.0,
                        help="extra random row dropout robustness check")
    parser.add_argument("--model", default=None)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    cfg = load_config()
    model_path = Path(args.model or Path(cfg["paths"]["models"]) / "panel_anfis.json")
    if not model_path.exists():
        raise SystemExit(f"model not found at {model_path} -- run train_anfis_panel.py first")
    model = TSKFuzzySystem.load(model_path)

    report = run(cfg, model, args.alpha, args.mask_rate, args.seed)
    logs = Path(cfg["paths"]["logs"])
    logs.mkdir(parents=True, exist_ok=True)
    metrics_path = logs / "svr_reconstruction_metrics.json"
    history = json.loads(metrics_path.read_text()) if metrics_path.exists() else []
    history = [h for h in history if not (h["alpha"] == args.alpha and h["mask_rate"] == args.mask_rate)]
    history.append(report)
    metrics_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    pd.DataFrame(history).to_csv(logs / "svr_reconstruction_metrics.csv", index=False)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
