"""Load the UCI Student Performance dataset (Math + Portuguese)."""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd
import urllib.request

from .config import DATA_RAW

UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip"


def download_if_missing() -> None:
    mat = DATA_RAW / "student-mat.csv"
    por = DATA_RAW / "student-por.csv"
    if mat.exists() and por.exists():
        return
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    print(f"Downloading UCI Student Performance dataset from {UCI_URL} ...")
    with urllib.request.urlopen(UCI_URL) as r:
        buf = io.BytesIO(r.read())
    with zipfile.ZipFile(buf) as z:
        z.extractall(DATA_RAW)
    print("Download complete.")


def load_raw(course: str = "both") -> pd.DataFrame:
    """Load raw dataset.

    course: 'mat' (math), 'por' (Portuguese), or 'both' (concatenated with a 'course' column).
    """
    download_if_missing()
    mat = pd.read_csv(DATA_RAW / "student-mat.csv", sep=";")
    por = pd.read_csv(DATA_RAW / "student-por.csv", sep=";")
    if course == "mat":
        return mat.assign(course="math")
    if course == "por":
        return por.assign(course="portuguese")
    mat["course"] = "math"
    por["course"] = "portuguese"
    return pd.concat([mat, por], ignore_index=True)


if __name__ == "__main__":
    df = load_raw("both")
    print(df.shape)
    print(df.head())
