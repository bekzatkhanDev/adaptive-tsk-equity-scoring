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
    val_loss: list[float] = field(default_factory=list)

    def to_frame(self) -> pd.DataFrame:
        """Per-epoch diagnostics as a DataFrame.

        ``val_loss`` is only recorded when early stopping is enabled, so the
        shorter series are aligned to ``train_loss`` by padding with ``NaN``
        rather than letting the DataFrame constructor reject unequal lengths.
        """
        n = len(self.train_loss)

        def align(values: list[float]) -> list[float]:
            padded = list(values)[:n]
            return padded + [np.nan] * (n - len(padded))

        return pd.DataFrame(
            {
                "epoch": np.arange(n, dtype=int),
                "train_loss": align(self.train_loss),
                "gradient_norm": align(self.gradient_norm),
                "learning_rate": align(self.learning_rate),
                "val_loss": align(self.val_loss),
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
    @staticmethod
    def _chronological_split(X: np.ndarray, y: np.ndarray, frac: float):
        """Split rows into (fit, holdout) blocks preserving panel order.

        The training panel is ordered ``ticker-major`` then date-ascending, so a
        contiguous tail split holds out each ticker's most recent dates -- a
        proper chronological validation block with no random shuffling.
        """
        n = len(X)
        n_fit = max(1, int(round(n * frac)))
        if n_fit >= n or n < 20:
            return X, y, None, None
        return X[:n_fit], y[:n_fit], X[n_fit:], y[n_fit:]

    def fit(self, X, y, epochs: int = 200, verbose_every: int = 25,
            early_stopping: bool = False, patience: int = 25,
            val_frac: float = 0.2) -> TrainingHistory:
        """Hybrid ANFIS training.

        With ``early_stopping=False`` (default) the run is bit-for-bit the
        canonical protocol: every epoch solves the consequents on all rows and
        takes one full-batch width step. With ``early_stopping=True`` the last
        ``val_frac`` of the (chronologically ordered) rows are held out as a
        validation block, model states are checkpointed every epoch, and
        training stops after ``patience`` epochs without validation-loss
        improvement; the best-checkpointed state is restored at the end. This
        guards against overfitting when Tier-2 feature vectors grow large
        relative to the sample.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        keep = np.isfinite(X).all(axis=1) & np.isfinite(y)
        X, y = X[keep], y[keep]

        X_fit, y_fit, X_val, y_val = X, y, None, None
        if early_stopping:
            X_fit, y_fit, X_val, y_val = self._chronological_split(X, y, 1.0 - val_frac)

        self.model.init_from_data(X) if self.model.centers.trace() == 0 else None

        def _snapshot():
            return {
                "sigmas": self.model.sigmas.copy(),
                "consequents": self.model.consequents.copy(),
            }

        def _restore(state):
            self.model.sigmas = state["sigmas"].copy()
            self.model.consequents = state["consequents"].copy()

        best_state, best_val, best_epoch, bad_epochs = None, np.inf, -1, 0
        for epoch in range(epochs):
            # Forward pass: refit consequents on the current premises.
            self.model.fit_consequents(X_fit, y_fit, ridge=self.ridge)
            # Backward pass: width gradient step (log-sigma parameterisation).
            grads = self.model.premise_gradients(X_fit, y_fit)["log_sigma"]
            norm = float(np.linalg.norm(grads))
            if norm > self.grad_clip:
                grads *= self.grad_clip / norm
            self.model.sigmas = np.maximum(
                np.exp(np.log(np.maximum(self.model.sigmas, 1e-12)) - self.lr * grads),
                self.model.sigma_floor,
            )
            loss = float(0.5 * np.sum((self.model.predict(X_fit) - y_fit) ** 2))
            self.history.train_loss.append(loss)
            self.history.gradient_norm.append(norm)
            self.history.learning_rate.append(self.lr)

            if X_val is not None:
                # Consequents were just solved on the fit block, so the current
                # state is a valid checkpoint candidate.
                val = float(0.5 * np.sum((self.model.predict(X_val) - y_val) ** 2))
                self.history.val_loss.append(val)
                if val < best_val - 1e-9:
                    best_val, best_epoch, bad_epochs = val, epoch, 0
                    best_state = _snapshot()
                else:
                    bad_epochs += 1
                    if bad_epochs >= patience:
                        if verbose_every:
                            print(f"early stop at epoch {epoch} "
                                  f"(best epoch {best_epoch}, val {best_val:.4f})")
                        break
            elif verbose_every and epoch % verbose_every == 0:
                print(f"epoch {epoch:4d}  loss={loss:.4f}  |g|={norm:.4f}")

        if X_val is not None and best_state is not None:
            _restore(best_state)
            # Refit consequents once more on ALL rows at the selected premises,
            # so the deployed model uses the full training sample (the standard
            # refit-on-full-data convention for early-stopped models).
            self.model.fit_consequents(X, y, ridge=self.ridge)

        # Final consequent refit on the refined widths.
        self.model.fit_consequents(X_fit, y_fit, ridge=self.ridge)
        self.model.training_metadata.update(
            {
                "n_train": int(len(X_fit)),
                "n_epochs": len(self.history.train_loss),
                "final_loss": self.history.train_loss[-1] if self.history.train_loss else None,
                "early_stopping": bool(early_stopping),
                "best_epoch": best_epoch if X_val is not None else None,
                "best_val_loss": best_val if X_val is not None else None,
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