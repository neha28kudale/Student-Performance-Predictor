"""Domain-driven feature engineering for the Student Performance dataset."""
from __future__ import annotations

import numpy as np
import pandas as pd


JOB_SES_MAP = {
    "at_home": 1,
    "other": 2,
    "services": 3,
    "teacher": 4,
    "health": 5,
}


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add domain-driven features. Returns a new DataFrame."""
    out = df.copy()

    # Attendance ratio (higher = better attendance). Cap absences at observed max.
    max_abs = max(out["absences"].max(), 1)
    out["attendance_ratio"] = 1.0 - (out["absences"] / max_abs)

    # Study hours grouping (UCI codes 1..4)
    study_map = {1: "Low", 2: "Medium", 3: "High", 4: "Very High"}
    out["study_hours_group"] = out["studytime"].map(study_map).fillna("Medium")

    # Parent education impact
    out["parent_edu_max"] = out[["Medu", "Fedu"]].max(axis=1)
    out["parent_edu_avg"] = out[["Medu", "Fedu"]].mean(axis=1)

    # Socio-economic index: parent edu + job tier + internet + family size
    famsize_score = (out["famsize"] == "LE3").astype(int)  # smaller family -> +1
    internet_score = (out["internet"] == "yes").astype(int)
    mjob_score = out["Mjob"].map(JOB_SES_MAP).fillna(2)
    fjob_score = out["Fjob"].map(JOB_SES_MAP).fillna(2)
    out["ses_index"] = (
        0.35 * out["parent_edu_avg"]
        + 0.25 * ((mjob_score + fjob_score) / 2.0)
        + 0.20 * internet_score
        + 0.10 * famsize_score
        + 0.10 * (out["Pstatus"] == "T").astype(int)
    )

    # Support score: school + family + paid tutoring + extracurriculars
    for c in ("schoolsup", "famsup", "paid", "activities"):
        out[f"_{c}_bin"] = (out[c] == "yes").astype(int)
    out["support_score"] = (
        out["_schoolsup_bin"]
        + out["_famsup_bin"]
        + out["_paid_bin"]
        + out["_activities_bin"]
    )
    out = out.drop(columns=[c for c in out.columns if c.startswith("_") and c.endswith("_bin")])

    # Lifestyle / risk
    out["alcohol_index"] = (out["Dalc"] + out["Walc"]) / 2.0
    out["risk_score"] = (
        out["failures"].clip(0, 4) * 1.0
        + out["alcohol_index"] * 0.5
        + (1 - out["attendance_ratio"]) * 2.0
    )

    return out


ENGINEERED_NUMERIC = [
    "attendance_ratio",
    "parent_edu_max",
    "parent_edu_avg",
    "ses_index",
    "support_score",
    "alcohol_index",
    "risk_score",
]
ENGINEERED_CATEGORICAL = ["study_hours_group"]
