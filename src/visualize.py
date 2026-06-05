"""Plot helpers: heatmap, feature importance, learning curves, residuals."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import learning_curve

from .config import FIGURES_DIR

sns.set_theme(style="whitegrid", context="talk")


def correlation_heatmap(df: pd.DataFrame, out: Path | None = None) -> Path:
    num = df.select_dtypes(include=[np.number])
    corr = num.corr()
    fig, ax = plt.subplots(figsize=(14, 11))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=False, ax=ax)
    ax.set_title("Numeric Feature Correlation Heatmap")
    out = out or (FIGURES_DIR / "correlation_heatmap.png")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def feature_importance_plot(names, importances, title: str, out: Path) -> Path:
    order = np.argsort(importances)[-20:]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(np.array(names)[order], np.array(importances)[order], color="#4C72B0")
    ax.set_title(title)
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def learning_curve_plot(estimator, X, y, title: str, out: Path, scoring=None) -> Path:
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=5, scoring=scoring, n_jobs=-1,
        train_sizes=np.linspace(0.2, 1.0, 5), random_state=42,
    )
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(train_sizes, train_scores.mean(axis=1), "o-", label="Train")
    ax.plot(train_sizes, val_scores.mean(axis=1), "o-", label="CV")
    ax.fill_between(train_sizes, train_scores.mean(1) - train_scores.std(1),
                    train_scores.mean(1) + train_scores.std(1), alpha=0.1)
    ax.fill_between(train_sizes, val_scores.mean(1) - val_scores.std(1),
                    val_scores.mean(1) + val_scores.std(1), alpha=0.1)
    ax.set_title(title)
    ax.set_xlabel("Training examples")
    ax.set_ylabel(f"Score ({scoring or 'default'})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def residual_plot(y_true, y_pred, out: Path) -> Path:
    resid = np.asarray(y_true) - np.asarray(y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].scatter(y_pred, resid, alpha=0.6, color="#4C72B0")
    axes[0].axhline(0, color="red", linestyle="--")
    axes[0].set_xlabel("Predicted G3")
    axes[0].set_ylabel("Residual")
    axes[0].set_title("Residuals vs Predictions")
    sns.histplot(resid, kde=True, ax=axes[1], color="#4C72B0")
    axes[1].set_title("Residual Distribution")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def prediction_distribution(y_true, y_pred, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.kdeplot(y_true, label="Actual", ax=ax, fill=True, alpha=0.4)
    sns.kdeplot(y_pred, label="Predicted", ax=ax, fill=True, alpha=0.4)
    ax.set_title("Prediction vs Actual Distribution (G3)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def confusion_plot(y_true, y_pred, labels, out: Path) -> Path:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Classification Confusion Matrix")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out
