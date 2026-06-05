"""Preprocessing: cleaning, encoding, scaling, splits."""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import (
    CLASS_BINS,
    CLASS_LABELS,
    DROP_LEAKAGE_FEATURES,
    LEAKAGE_COLS,
    RANDOM_STATE,
    TARGET_CLF,
    TARGET_REG,
    TEST_SIZE,
)
from .feature_engineering import engineer_features


def add_classification_target(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out[TARGET_CLF] = pd.cut(out[TARGET_REG], bins=CLASS_BINS, labels=CLASS_LABELS)
    return out


def split_features_target(
    df: pd.DataFrame, task: str
) -> Tuple[pd.DataFrame, pd.Series]:
    assert task in ("regression", "classification")
    target = TARGET_REG if task == "regression" else TARGET_CLF
    drop_cols = [TARGET_REG, TARGET_CLF]
    if DROP_LEAKAGE_FEATURES:
        drop_cols += LEAKAGE_COLS
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df[target]
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    # sklearn changed the kwarg name in 1.2; support both.
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", ohe, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def get_train_test(task: str, course: str = "both"):
    """End-to-end: load -> engineer -> add target -> split."""
    from .data_loader import load_raw

    df = load_raw(course=course)
    df = engineer_features(df)
    df = add_classification_target(df)
    X, y = split_features_target(df, task)
    stratify = y if task == "classification" else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=stratify
    )
    return X_train, X_test, y_train, y_test, df
