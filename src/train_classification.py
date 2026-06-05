"""Train classifiers to predict performance class (Excellent / Average / At Risk)."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:  # pragma: no cover
    from sklearn.ensemble import GradientBoostingClassifier as XGBClassifier  # type: ignore
    HAS_XGB = False

from .config import CLASS_LABELS, FIGURES_DIR, MODELS_DIR, RANDOM_STATE
from .evaluate import classification_metrics
from .preprocessing import build_preprocessor, get_train_test
from .visualize import confusion_plot, feature_importance_plot, learning_curve_plot


def _model_specs():
    specs = {
        "LogisticRegression": (
            LogisticRegression(max_iter=2000, class_weight="balanced"),
            {"model__C": [0.1, 1.0, 3.0]},
        ),
        "RandomForestClassifier": (
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced"),
            {
                "model__n_estimators": [200, 400],
                "model__max_depth": [None, 8, 16],
                "model__min_samples_leaf": [1, 2, 4],
            },
        ),
    }
    if HAS_XGB:
        specs["XGBClassifier"] = (
            XGBClassifier(
                random_state=RANDOM_STATE, n_jobs=-1,
                eval_metric="mlogloss", verbosity=0,
            ),
            {
                "model__n_estimators": [200, 400],
                "model__max_depth": [3, 5, 7],
                "model__learning_rate": [0.05, 0.1],
            },
        )
    else:
        specs["GradientBoostingClassifier"] = (
            XGBClassifier(random_state=RANDOM_STATE),
            {
                "model__n_estimators": [200, 400],
                "model__max_depth": [3, 5],
                "model__learning_rate": [0.05, 0.1],
            },
        )
    return specs


def run() -> dict:
    X_train, X_test, y_train, y_test, df_full = get_train_test("classification")

    # For XGBoost we need integer labels; map consistently.
    label_to_int = {l: i for i, l in enumerate(CLASS_LABELS)}
    int_to_label = {i: l for l, i in label_to_int.items()}
    y_train_i = y_train.map(label_to_int).astype(int)
    y_test_i = y_test.map(label_to_int).astype(int)

    pre = build_preprocessor(X_train)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    results = {}
    best_overall = (None, None, -np.inf)  # (name, pipeline, f1)

    for name, (model, grid) in _model_specs().items():
        pipe = Pipeline([("pre", pre), ("model", model)])
        gs = GridSearchCV(pipe, grid, cv=cv, scoring="f1_macro", n_jobs=-1)
        gs.fit(X_train, y_train_i)
        est = gs.best_estimator_
        preds_i = est.predict(X_test)
        preds = pd.Series(preds_i).map(int_to_label)
        metrics = classification_metrics(y_test_i.map(int_to_label), preds)
        results[name] = {"metrics": metrics, "best_params": gs.best_params_}
        print(f"[CLF] {name}: {metrics} params={gs.best_params_}")
        if metrics["F1_macro"] > best_overall[2]:
            best_overall = (name, est, metrics["F1_macro"])

    best_name, best_est, _ = best_overall
    joblib.dump(
        {"pipeline": best_est, "label_map": int_to_label},
        MODELS_DIR / "best_classifier.joblib",
    )

    # Diagnostics
    best_preds_i = best_est.predict(X_test)
    best_preds = pd.Series(best_preds_i).map(int_to_label)
    confusion_plot(
        y_test_i.map(int_to_label), best_preds, CLASS_LABELS,
        FIGURES_DIR / "confusion_matrix.png",
    )
    learning_curve_plot(
        best_est, X_train, y_train_i,
        title=f"Learning Curve — {best_name}",
        out=FIGURES_DIR / "learning_curve_classification.png",
        scoring="f1_macro",
    )

    try:
        feat_names = best_est.named_steps["pre"].get_feature_names_out()
        model = best_est.named_steps["model"]
        if hasattr(model, "feature_importances_"):
            feature_importance_plot(
                feat_names, model.feature_importances_,
                title=f"Feature Importance — {best_name}",
                out=FIGURES_DIR / "feature_importance_clf.png",
            )
        elif hasattr(model, "coef_"):
            imp = np.abs(model.coef_).mean(axis=0)
            feature_importance_plot(
                feat_names, imp,
                title=f"|Coefficients| — {best_name}",
                out=FIGURES_DIR / "feature_importance_clf.png",
            )
    except Exception as e:  # pragma: no cover
        print("Feature importance plot skipped:", e)

    summary = {
        "task": "classification",
        "target": "performance_class",
        "labels": CLASS_LABELS,
        "best_model": best_name,
        "results": results,
    }
    (MODELS_DIR / "classification_metrics.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    run()
