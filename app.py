"""Streamlit dashboard: Student Performance Predictor."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from src.config import CLASS_LABELS, FIGURES_DIR, MODELS_DIR
from src.data_loader import load_raw
from src.feature_engineering import engineer_features
from src.predict import predict_class, predict_score


st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
)

st.title("🎓 Student Performance Predictor")
st.caption("Regression + Classification ML system on the UCI Student Performance dataset.")


# ---------- Sidebar: student inputs ----------
st.sidebar.header("Student Profile")

def sb_select(label, options, index=0):
    return st.sidebar.selectbox(label, options, index=index)

def sb_slider(label, mn, mx, val, step=1):
    return st.sidebar.slider(label, mn, mx, val, step)

with st.sidebar:
    school = sb_select("School", ["GP", "MS"])
    sex = sb_select("Sex", ["F", "M"])
    age = sb_slider("Age", 15, 22, 17)
    address = sb_select("Address", ["U", "R"])
    famsize = sb_select("Family size", ["LE3", "GT3"])
    Pstatus = sb_select("Parent cohabitation", ["T", "A"])
    Medu = sb_slider("Mother education (0-4)", 0, 4, 3)
    Fedu = sb_slider("Father education (0-4)", 0, 4, 3)
    Mjob = sb_select("Mother job", ["teacher", "health", "services", "at_home", "other"])
    Fjob = sb_select("Father job", ["teacher", "health", "services", "at_home", "other"], index=2)
    reason = sb_select("School choice reason", ["home", "reputation", "course", "other"])
    guardian = sb_select("Guardian", ["mother", "father", "other"])
    traveltime = sb_slider("Travel time (1-4)", 1, 4, 1)
    studytime = sb_slider("Weekly study time (1-4)", 1, 4, 2)
    failures = sb_slider("Past failures", 0, 4, 0)
    schoolsup = sb_select("Extra school support", ["yes", "no"], index=1)
    famsup = sb_select("Family support", ["yes", "no"])
    paid = sb_select("Paid tutoring", ["yes", "no"], index=1)
    activities = sb_select("Extracurriculars", ["yes", "no"])
    nursery = sb_select("Attended nursery", ["yes", "no"])
    higher = sb_select("Wants higher education", ["yes", "no"])
    internet = sb_select("Internet at home", ["yes", "no"])
    romantic = sb_select("In a relationship", ["yes", "no"], index=1)
    famrel = sb_slider("Family relationship quality (1-5)", 1, 5, 4)
    freetime = sb_slider("Free time (1-5)", 1, 5, 3)
    goout = sb_slider("Going out (1-5)", 1, 5, 3)
    Dalc = sb_slider("Workday alcohol (1-5)", 1, 5, 1)
    Walc = sb_slider("Weekend alcohol (1-5)", 1, 5, 2)
    health = sb_slider("Health (1-5)", 1, 5, 4)
    absences = sb_slider("Absences", 0, 75, 4)
    course = sb_select("Course", ["math", "portuguese"])

raw = dict(
    school=school, sex=sex, age=age, address=address, famsize=famsize,
    Pstatus=Pstatus, Medu=Medu, Fedu=Fedu, Mjob=Mjob, Fjob=Fjob,
    reason=reason, guardian=guardian, traveltime=traveltime, studytime=studytime,
    failures=failures, schoolsup=schoolsup, famsup=famsup, paid=paid,
    activities=activities, nursery=nursery, higher=higher, internet=internet,
    romantic=romantic, famrel=famrel, freetime=freetime, goout=goout,
    Dalc=Dalc, Walc=Walc, health=health, absences=absences, course=course,
)

tab_pred, tab_analytics, tab_about = st.tabs(["🔮 Prediction", "📊 Analytics", "ℹ️ About"])

with tab_pred:
    col1, col2 = st.columns([1, 1])

    try:
        score = predict_score(raw)
        clf = predict_class(raw)
    except FileNotFoundError:
        st.error("Trained models not found in `models/`. Run "
                 "`python -m src.train_regression && python -m src.train_classification` first.")
        st.stop()

    with col1:
        st.subheader("Predicted Final Score (G3)")
        st.metric("G3 (0–20)", f"{score:.2f}")
        st.progress(min(max(score / 20.0, 0.0), 1.0))

    with col2:
        st.subheader("Performance Class")
        color = {"Excellent": "🟢", "Average": "🟡", "At Risk": "🔴"}.get(clf["label"], "⚪️")
        st.markdown(f"## {color} **{clf['label']}**")
        if "proba" in clf:
            proba_df = (
                pd.DataFrame({"class": list(clf["proba"].keys()),
                              "confidence": list(clf["proba"].values())})
                .set_index("class")
            )
            st.bar_chart(proba_df)
            top = max(clf["proba"], key=clf["proba"].get)
            st.caption(f"Confidence: **{clf['proba'][top]*100:.1f}%** for class **{top}**")

    st.divider()
    st.subheader("Top Feature Importances (best model)")
    fi = FIGURES_DIR / "feature_importance_clf.png"
    if fi.exists():
        st.image(str(fi))
    else:
        st.info("Run training to generate the feature-importance chart.")

with tab_analytics:
    st.subheader("Dataset Overview")
    df = engineer_features(load_raw("both"))
    c1, c2, c3 = st.columns(3)
    c1.metric("Students", len(df))
    c2.metric("Avg final score", f"{df['G3'].mean():.2f}")
    c3.metric("Avg absences", f"{df['absences'].mean():.1f}")

    st.bar_chart(df["G3"].value_counts().sort_index())

    st.subheader("Model Metrics")
    reg_p = MODELS_DIR / "regression_metrics.json"
    clf_p = MODELS_DIR / "classification_metrics.json"
    if reg_p.exists():
        reg = json.loads(reg_p.read_text())
        st.markdown("**Regression**")
        st.dataframe(pd.DataFrame({k: v["metrics"] for k, v in reg["results"].items()}).T)
        st.caption(f"Best model: **{reg['best_model']}**")
    if clf_p.exists():
        clf_m = json.loads(clf_p.read_text())
        st.markdown("**Classification**")
        st.dataframe(pd.DataFrame({k: v["metrics"] for k, v in clf_m["results"].items()}).T)
        st.caption(f"Best model: **{clf_m['best_model']}**")

    st.subheader("Diagnostics")
    for name in ("correlation_heatmap.png", "confusion_matrix.png", "residuals.png",
                 "prediction_distribution.png", "feature_importance_reg.png",
                 "feature_importance_clf.png", "learning_curve_regression.png",
                 "learning_curve_classification.png"):
        p = FIGURES_DIR / name
        if p.exists():
            st.image(str(p), caption=name)

with tab_about:
    st.markdown("""
**Student Performance Predictor** trains regression and classification models on the
UCI Student Performance dataset (Cortez & Silva, 2008) to predict final exam score (G3)
and a 3-class performance label (Excellent / Average / At Risk).

- Models: Linear/Logistic Regression, Random Forest, XGBoost
- Tuning: GridSearchCV with 5-fold (Stratified)KFold
- Feature engineering: attendance ratio, study-hours grouping, parent-education impact, SES index, support score, risk score
- Built with scikit-learn, XGBoost, pandas, Streamlit.

See `README.md` for the full project documentation.
""")
