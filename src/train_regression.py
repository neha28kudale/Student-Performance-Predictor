"""Train regression models to predict final exam score (G3).

Compares Linear Regression, Random Forest, and XGBoost with GridSearchCV
and saves the best model + metrics + diagnostic plots.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:  # pragma: no cover
    from sklearn.ensemble import GradientBoostingRegressor as XGBRegressor  # type: ignore
    HAS_XGB = False

from .config import FIGURES_DIR, MODELS_DIR, RANDOM_STATE
from .evaluate import regression_metrics
from .preprocessing import build_preprocessor, get_train_test
from .visualize import (
    feature_importance_plot,
    learning_curve_plot,
    prediction_distribution,
    residual_plot,
)


def _model_specs():
    specs = {
        "LinearRegression": (LinearRegression(), {}),
        "RandomForestRegressor": (
            RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
            {
                "model__n_estimators": [200, 400],
                "model__max_depth": [None, 8, 16],
                "model__min_samples_split": [2, 5],
            },
        ),
    }
    if HAS_XGB:
        specs["XGBRegressor"] = (
            XGBRegressor(
                random_state=RANDOM_STATE, n_jobs=-1,
                objective="reg:squarederror", verbosity=0,
            ),
            {
                "model__n_estimators": [200, 400],
                "model__max_depth": [3, 5, 7],
                "model__learning_rate": [0.05, 0.1],
                "model__subsample": [0.8, 1.0],
            },
        )
    else:
        specs["GradientBoostingRegressor"] = (
            XGBRegressor(random_state=RANDOM_STATE),
            {
                "model__n_estimators": [200, 400],
                "model__max_depth": [3, 5],
                "model__learning_rate": [0.05, 0.1],
            },
        )
    return specs


def run() -> dict:
    X_train, X_test, y_train, y_test, df_full = get_train_test("regression")
    pre = build_preprocessor(X_train)

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = {}
    best_overall = (None, None, np.inf)  # (name, pipeline, rmse)

    for name, (model, grid) in _model_specs().items():
        pipe = Pipeline([("pre", pre), ("model", model)])
        if grid:
            gs = GridSearchCV(pipe, grid, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1)
            gs.fit(X_train, y_train)
            est = gs.best_estimator_
            best_params = gs.best_params_
        else:
            pipe.fit(X_train, y_train)
            est = pipe
            best_params = {}
        preds = est.predict(X_test)
        metrics = regression_metrics(y_test, preds)
        results[name] = {"metrics": metrics, "best_params": best_params}
        print(f"[REG] {name}: {metrics} params={best_params}")
        if metrics["RMSE"] < best_overall[2]:
            best_overall = (name, est, metrics["RMSE"])

    best_name, best_est, _ = best_overall
    joblib.dump(best_est, MODELS_DIR / "best_regressor.joblib")

    # Diagnostics on the best model
    best_preds = best_est.predict(X_test)
    residual_plot(y_test, best_preds, FIGURES_DIR / "residuals.png")
    prediction_distribution(y_test, best_preds, FIGURES_DIR / "prediction_distribution.png")
    learning_curve_plot(
        best_est, X_train, y_train,
        title=f"Learning Curve — {best_name}",
        out=FIGURES_DIR / f"learning_curve_regression.png",
        scoring="neg_root_mean_squared_error",
    )

    # Feature importance (post-preprocessing names)
    try:
        feat_names = best_est.named_steps["pre"].get_feature_names_out()
        model = best_est.named_steps["model"]
        if hasattr(model, "feature_importances_"):
            feature_importance_plot(
                feat_names, model.feature_importances_,
                title=f"Feature Importance — {best_name}",
                out=FIGURES_DIR / "feature_importance_reg.png",
            )
        elif hasattr(model, "coef_"):
            feature_importance_plot(
                feat_names, np.abs(model.coef_),
                title=f"|Coefficients| — {best_name}",
                out=FIGURES_DIR / "feature_importance_reg.png",
            )
    except Exception as e:  # pragma: no cover
        print("Feature importance plot skipped:", e)

    summary = {
        "task": "regression",
        "target": "G3",
        "best_model": best_name,
        "results": results,
    }
    (MODELS_DIR / "regression_metrics.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    run()
