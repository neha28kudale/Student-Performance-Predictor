"""Inference helpers used by the Streamlit app and CLI."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd

from .config import MODELS_DIR
from .feature_engineering import engineer_features


def _ensure_features(df: pd.DataFrame) -> pd.DataFrame:
    needed = {"attendance_ratio", "ses_index", "support_score"}
    if not needed.issubset(df.columns):
        df = engineer_features(df)
    drop = [c for c in ("G3", "G1", "G2", "performance_class") if c in df.columns]
    return df.drop(columns=drop)


def predict_score(raw_input: Dict | pd.DataFrame) -> float:
    pipe = joblib.load(MODELS_DIR / "best_regressor.joblib")
    df = pd.DataFrame([raw_input]) if isinstance(raw_input, dict) else raw_input.copy()
    X = _ensure_features(df)
    pred = float(np.clip(pipe.predict(X)[0], 0, 20))
    return pred


def predict_class(raw_input: Dict | pd.DataFrame) -> Dict:
    bundle = joblib.load(MODELS_DIR / "best_classifier.joblib")
    pipe = bundle["pipeline"]
    int_to_label = bundle["label_map"]
    df = pd.DataFrame([raw_input]) if isinstance(raw_input, dict) else raw_input.copy()
    X = _ensure_features(df)
    pred_i = int(pipe.predict(X)[0])
    label = int_to_label[pred_i]
    out = {"label": label, "label_index": pred_i}
    if hasattr(pipe.named_steps["model"], "predict_proba"):
        proba = pipe.predict_proba(X)[0]
        out["proba"] = {int_to_label[i]: float(p) for i, p in enumerate(proba)}
    return out


if __name__ == "__main__":
    sample = {
        "school": "GP", "sex": "F", "age": 17, "address": "U", "famsize": "GT3",
        "Pstatus": "T", "Medu": 4, "Fedu": 4, "Mjob": "teacher", "Fjob": "services",
        "reason": "course", "guardian": "mother", "traveltime": 1, "studytime": 3,
        "failures": 0, "schoolsup": "no", "famsup": "yes", "paid": "no",
        "activities": "yes", "nursery": "yes", "higher": "yes", "internet": "yes",
        "romantic": "no", "famrel": 4, "freetime": 3, "goout": 2, "Dalc": 1,
        "Walc": 1, "health": 5, "absences": 2, "course": "math",
    }
    print("Score:", predict_score(sample))
    print("Class:", predict_class(sample))
