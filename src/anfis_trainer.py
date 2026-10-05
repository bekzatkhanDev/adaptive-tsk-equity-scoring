"""ANFIS hybrid training (Section 3.3.1 of the paper).

Each epoch runs the two-pass hybrid paradigm:

1. **Forward pass** -- consequents solved by ridge least squares (Eq. (9)) on
   the current premises;
2. **Backward pass** -- Gaussian widths updated by gradient descent on
   ``E = 0.5 * sum (Y_raw - Y_target)^2`` with learning rate ``eta = 1e-3``
   (centres stay cluster-fixed).

The loss reported is computed on the raw (unclamped) defuzzified output so the
gradient and the objective coincide.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from src.TSK_engine import TSKFuzzySystem

__all__ = ["ANFISTrainer", "TrainingHistory"]


@dataclass
class TrainingHistory:
    """Per-epoch diagnostics of an ANFIS run."""

    train_loss: list[float] = field(default_factory=list)
    gradient_norm: list[float] = field(default_factory=list)
    learning_rate: list[float] = field(default_factory=list)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "epoch": np.arange(len(self.train_loss), dtype=int),
                "train_loss": self.train_loss,
                "gradient_norm": self.gradient_norm,
                "learning_rate": self.learning_rate,
            }
        )


class ANFISTrainer:
    """Hybrid-trained TSK system (widths-only backward pass)."""

    def __init__(
        self,
        model: TSKFuzzySystem,
        lr: float = 1e-3,
        ridge: float = 1e-6,
        grad_clip: float = 5.0,
    ) -> None:
        self.model = model
        self.lr = float(lr)
        self.ridge = float(ridge)
        self.grad_clip = float(grad_clip)
        self.history = TrainingHistory()

    # ------------------------------------------------------------------ fit
    def fit(self, X, y, epochs: int = 200, verbose_every: int = 25) -> TrainingHistory:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        keep = np.isfinite(X).all(axis=1) & np.isfinite(y)
        X, y = X[keep], y[keep]

        self.model.init_from_data(X) if self.model.centers.trace() == 0 else None
        for epoch in range(epochs):
            # Forward pass: refit consequents on the current premises.
            self.model.fit_consequents(X, y, ridge=self.ridge)
            # Backward pass: width gradient step (log-sigma parameterisation).
            grads = self.model.premise_gradients(X, y)["log_sigma"]
            norm = float(np.linalg.norm(grads))
            if norm > self.grad_clip:
                grads *= self.grad_clip / norm
            self.model.sigmas = np.maximum(
                np.exp(np.log(np.maximum(self.model.sigmas, 1e-12)) - self.lr * grads),
                self.model.sigma_floor,
            )
            loss = float(0.5 * np.sum((self.model.predict(X) - y) ** 2))
            self.history.train_loss.append(loss)
            self.history.gradient_norm.append(norm)
            self.history.learning_rate.append(self.lr)
            if verbose_every and epoch % verbose_every == 0:
                print(f"epoch {epoch:4d}  loss={loss:.4f}  |g|={norm:.4f}")

        # Final consequent refit on the refined widths.
        self.model.fit_consequents(X, y, ridge=self.ridge)
        self.model.training_metadata.update(
            {
                "n_train": int(len(X)),
                "n_epochs": len(self.history.train_loss),
                "final_loss": self.history.train_loss[-1] if self.history.train_loss else None,
            }
        )
        return self.history

    # ------------------------------------------------------------ evaluation
    def evaluate(self, X, y) -> dict[str, float]:
        """RMSE / MAE / directional accuracy / Rank IC on a sample."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        keep = np.isfinite(X).all(axis=1) & np.isfinite(y)
        X, y = X[keep], y[keep]
        prediction = self.model.predict(X)
        from scipy import stats

        ic = float("nan")
        if len(y) >= 3 and np.std(y) > 0 and np.std(prediction) > 0:
            ic = float(stats.spearmanr(prediction, y).statistic)
        return {
            "n": float(len(y)),
            "rmse": float(np.sqrt(np.mean((prediction - y) ** 2))),
            "mae": float(np.mean(np.abs(prediction - y))),
            "dir_acc": float(np.mean(np.sign(prediction - 50.0) == np.sign(y - 50.0))),
            "rank_ic": ic,
        }

    # ----------------------------------------------------------- persistence
    def save_checkpoint(self, path, extra: Mapping | None = None) -> Path:
        payload = {
            "model": self.model.to_dict(),
            "history": self.history.to_frame().to_dict(orient="list"),
            "extra": dict(extra or {}),
        }
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return target

    @classmethod
    def load_checkpoint(cls, path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        model = TSKFuzzySystem.from_dict(payload["model"])
        trainer = cls(model)
        history = payload.get("history", {})
        trainer.history = TrainingHistory(
            train_loss=list(history.get("train_loss", [])),
            gradient_norm=list(history.get("gradient_norm", [])),
            learning_rate=list(history.get("learning_rate", [])),
        )
        return trainer, dict(payload.get("extra", {}))