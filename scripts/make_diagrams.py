"""Generate architecture diagram and mock dashboard screenshots for the README."""
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT = Path(__file__).parent.parent
FIG = ROOT / "reports" / "figures"
SCR = FIG / "screenshots"
SCR.mkdir(parents=True, exist_ok=True)
ARCH = ROOT / "reports" / "architecture.png"


def architecture():
    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.set_xlim(0, 13); ax.set_ylim(0, 6.5); ax.axis("off")

    nodes = [
        (0.5, 2.5, 2.2, 1.6, "UCI Student\nPerformance\nDataset", "#E8F0FE"),
        (3.2, 2.5, 2.2, 1.6, "Preprocessing\n• cleaning\n• encoding\n• scaling", "#FCE8E6"),
        (5.9, 2.5, 2.2, 1.6, "Feature\nEngineering\n• attendance\n• SES, support", "#E6F4EA"),
        (8.6, 4.5, 2.2, 1.6, "Regression\nLinear / RF /\nXGBoost", "#FEF7E0"),
        (8.6, 0.5, 2.2, 1.6, "Classification\nLogReg / RF /\nXGBoost", "#FEF7E0"),
        (11.3, 2.5, 1.5, 1.6, "Model\nRegistry\n(joblib)", "#F3E8FD"),
    ]
    for x, y, w, h, text, color in nodes:
        ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                                             facecolor=color, edgecolor="#444"))
        ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=10)

    # Arrows
    arrows = [(2.7, 3.3, 0.5, 0), (5.4, 3.3, 0.5, 0),
              (8.1, 3.3, 0.5, 2.0), (8.1, 3.3, 0.5, -2.0),
              (10.8, 5.3, 0.5, -2.0), (10.8, 1.3, 0.5, 2.0)]
    for x, y, dx, dy in arrows:
        ax.annotate("", xy=(x + dx, y + dy), xytext=(x, y),
                    arrowprops=dict(arrowstyle="->", color="#444", lw=1.8))

    # Streamlit consumer
    ax.add_patch(mpatches.FancyBboxPatch((5.0, 5.3), 3.0, 1.0,
                                         boxstyle="round,pad=0.05",
                                         facecolor="#D2E3FC", edgecolor="#444"))
    ax.text(6.5, 5.8, "Streamlit Dashboard\n(app.py)", ha="center", va="center", fontsize=10)
    ax.annotate("", xy=(11.3, 4.0), xytext=(8.0, 5.5),
                arrowprops=dict(arrowstyle="<-", color="#444", lw=1.5, ls="--"))

    ax.set_title("Student Performance Predictor — System Architecture", fontsize=14, pad=10)
    fig.tight_layout()
    fig.savefig(ARCH, dpi=140)
    plt.close(fig)


def _frame(ax, title, color="#1f6feb"):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor("#d0d7de")
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=color, pad=8)


def screenshot_prediction():
    fig = plt.figure(figsize=(12, 7), facecolor="white")
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1.4, 1.4], hspace=0.55, wspace=0.35)

    # Sidebar (inputs)
    ax0 = fig.add_subplot(gs[:, 0]); _frame(ax0, "Student Profile")
    items = [("School", "GP"), ("Sex", "F"), ("Age", "17"),
             ("Mother edu", "4"), ("Father edu", "4"),
             ("Study time", "3"), ("Failures", "0"),
             ("Absences", "4"), ("Internet", "yes"),
             ("Activities", "yes"), ("Higher ed", "yes")]
    for i, (k, v) in enumerate(items):
        ax0.text(0.05, 0.95 - i*0.085, f"{k}", fontsize=9, color="#57606a")
        ax0.text(0.95, 0.95 - i*0.085, v, fontsize=10, color="#0d1117",
                 ha="right", fontweight="bold")

    # Predicted score
    ax1 = fig.add_subplot(gs[0, 1]); _frame(ax1, "Predicted Final Score (G3)")
    ax1.text(0.5, 0.65, "14.20", ha="center", fontsize=42, fontweight="bold", color="#1f6feb")
    ax1.text(0.5, 0.30, "out of 20", ha="center", fontsize=10, color="#57606a")
    ax1.add_patch(mpatches.Rectangle((0.1, 0.10), 0.8, 0.06, color="#d0d7de"))
    ax1.add_patch(mpatches.Rectangle((0.1, 0.10), 0.8*0.71, 0.06, color="#1f6feb"))

    # Performance class
    ax2 = fig.add_subplot(gs[0, 2]); _frame(ax2, "Performance Class")
    ax2.text(0.5, 0.62, "🟢  Average", ha="center", fontsize=22, fontweight="bold", color="#1a7f37")
    ax2.text(0.5, 0.32, "Confidence: 62.4%", ha="center", fontsize=11, color="#57606a")

    # Confidence chart
    ax3 = fig.add_subplot(gs[1, 1]); _frame(ax3, "Class Probabilities")
    classes = ["At Risk", "Average", "Excellent"]
    probs = [0.18, 0.62, 0.20]
    bars = ax3.barh(classes, probs, color=["#cf222e", "#1f6feb", "#1a7f37"])
    ax3.set_xlim(0, 1); ax3.invert_yaxis()
    for b, p in zip(bars, probs):
        ax3.text(p + 0.02, b.get_y() + b.get_height()/2, f"{p*100:.0f}%",
                 va="center", fontsize=9)

    # Feature importance preview
    ax4 = fig.add_subplot(gs[1, 2]); _frame(ax4, "Top Feature Importances")
    feats = ["absences", "failures", "ses_index", "studytime", "attendance_ratio", "higher_yes"]
    vals = [0.21, 0.18, 0.12, 0.10, 0.09, 0.07]
    ax4.barh(feats[::-1], vals[::-1], color="#1f6feb")
    ax4.set_xlim(0, 0.25)

    fig.suptitle("🎓  Student Performance Predictor — Prediction Tab",
                 fontsize=15, fontweight="bold", y=0.98, x=0.06, ha="left")
    fig.savefig(SCR / "01_prediction_tab.png", dpi=140, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


def screenshot_analytics():
    fig = plt.figure(figsize=(12, 7), facecolor="white")
    gs = fig.add_gridspec(2, 3, hspace=0.55, wspace=0.35)

    ax0 = fig.add_subplot(gs[0, 0]); _frame(ax0, "Students")
    ax0.text(0.5, 0.5, "1,044", ha="center", fontsize=38, fontweight="bold", color="#1f6feb")

    ax1 = fig.add_subplot(gs[0, 1]); _frame(ax1, "Avg Final Score")
    ax1.text(0.5, 0.5, "11.34", ha="center", fontsize=38, fontweight="bold", color="#1f6feb")

    ax2 = fig.add_subplot(gs[0, 2]); _frame(ax2, "Avg Absences")
    ax2.text(0.5, 0.5, "4.4", ha="center", fontsize=38, fontweight="bold", color="#1f6feb")

    ax3 = fig.add_subplot(gs[1, :2]); _frame(ax3, "G3 Score Distribution")
    rng = np.random.default_rng(7)
    data = np.clip(rng.normal(11.3, 3.2, 1044), 0, 20).astype(int)
    ax3.hist(data, bins=21, color="#1f6feb", edgecolor="white")

    ax4 = fig.add_subplot(gs[1, 2]); _frame(ax4, "Model Leaderboard (F1)")
    ax4.barh(["LogReg", "RandomForest", "XGBoost"], [0.43, 0.49, 0.46],
             color=["#8250df", "#1a7f37", "#bf8700"])
    ax4.set_xlim(0, 0.6)

    fig.suptitle("📊  Analytics Tab", fontsize=15, fontweight="bold",
                 y=0.98, x=0.06, ha="left")
    fig.savefig(SCR / "02_analytics_tab.png", dpi=140, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    architecture()
    screenshot_prediction()
    screenshot_analytics()
    print("Architecture + mock screenshots generated.")
