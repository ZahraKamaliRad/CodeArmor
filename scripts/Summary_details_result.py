import os
import re
import pandas as pd
from decimal import Decimal
from openpyxl import load_workbook
from openpyxl.utils import get_column_letter
from openpyxl.styles import Alignment
import matplotlib.pyplot as plt
import numpy as np

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_PATH)
BASE_DIR = os.path.join(PROJECT_ROOT, "outputs")
OUTPUT_FILE = os.path.join(BASE_PATH, "summary.xlsx")
CHARTS_DIR = os.path.join(BASE_PATH, "charts")

os.makedirs(CHARTS_DIR, exist_ok=True)


def autosize_columns(ws):
    for column in ws.columns:
        max_length = 0
        col = column[0].column
        for cell in column:
            if cell.value:
                max_length = max(max_length, len(str(cell.value)))
        ws.column_dimensions[get_column_letter(col)].width = max_length + 2


def center_align(ws):
    align = Alignment(horizontal="center", vertical="center")
    for row in ws.iter_rows():
        for cell in row:
            cell.alignment = align


def count_decimals(value):
    try:
        d = Decimal(str(value))
        return max(0, -d.as_tuple().exponent)
    except:
        return 0


try:
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)
except PermissionError:
    pass


dataframes = []

for root, _, files in os.walk(BASE_DIR):
    for file in files:
        if not file.endswith(".xlsx") or file.startswith("~$"):
            continue
        df = pd.read_excel(os.path.join(root, file))
        dataframes.append(df)

all_data = pd.concat(dataframes, ignore_index=True)

all_data["technique"] = all_data["technique"].astype(str).str.replace(r"\(\d+\)$", "", regex=True)

group_cols = ["dataset", "model", "technique"]
summary_rows = []

for _, group in all_data.groupby(group_cols):

    numeric_cols = group.select_dtypes(include="number").columns

    decimal_map = {}
    for col in numeric_cols:
        first_vals = group[col].dropna()
        if len(first_vals) == 0:
            decimal_map[col] = 2
        else:
            decimal_map[col] = count_decimals(first_vals.iloc[0])

    means = group[numeric_cols].mean()

    for col in numeric_cols:
        means[col] = round(means[col], decimal_map.get(col, 2))

    base_row = group.iloc[0].copy()
    for col in numeric_cols:
        base_row[col] = means[col]

    summary_rows.append(base_row)

summary_df = pd.DataFrame(summary_rows)
summary_df = summary_df[all_data.columns]
summary_df = summary_df.sort_values(by=["dataset", "model", "technique"]).reset_index(drop=True)

summary_df["bandit_rate"] = summary_df["bandit_rate"].round(2)
summary_df["bandit_density"] = summary_df["bandit_density"].round(2)

reduction_rows = []

for (dataset, model), group in summary_df.groupby(["dataset", "model"]):

    direct = group[group["technique"] == "direct"]
    if direct.empty:
        continue

    direct = direct.iloc[0]

    d_bandit_rate = direct.get("bandit_rate", 0)
    d_bandit_density = direct.get("bandit_density", 0)
    d_semgrep_rate = direct.get("semgrep_rate", 0)
    d_semgrep_density = direct.get("semgrep_density", 0)

    for _, row in group.iterrows():

        def safe_reduction(base, value):
            if pd.isna(base) or base == 0:
                return 0
            return ((base - value) / base) * 100

        reduction_rows.append({
            "dataset": row["dataset"],
            "model": row["model"],
            "technique": row["technique"],
            "bandit_rate(*100)": row.get("bandit_rate", 0) * 100,
            "bandit_density(*1000)": row.get("bandit_density", 0) * 1000,
            "semgrep_rate(*100)": row.get("semgrep_rate", 0) * 100,
            "semgrep_density(*1000)": row.get("semgrep_density", 0) * 1000,
            "%▼bandit_rate": round(safe_reduction(d_bandit_rate, row.get("bandit_rate", 0)), 2),
            "%▼bandit_density": round(safe_reduction(d_bandit_density, row.get("bandit_density", 0)), 2),
            "%▼semgrep_rate": round(safe_reduction(d_semgrep_rate, row.get("semgrep_rate", 0)), 2),
            "%▼semgrep_density": round(safe_reduction(d_semgrep_density, row.get("semgrep_density", 0)), 2),
        })

reduction_df = pd.DataFrame(reduction_rows)

reduction_df = reduction_df[
    [
        "dataset", "model", "technique",
        "bandit_rate(*100)", "bandit_density(*1000)",
        "semgrep_rate(*100)", "semgrep_density(*1000)",
        "%▼bandit_rate", "%▼bandit_density",
        "%▼semgrep_rate", "%▼semgrep_density"
    ]
]

reduction_df = reduction_df.sort_values(by=["dataset", "model", "technique"]).reset_index(drop=True)

with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
    summary_df.to_excel(writer, sheet_name="summary", index=False)
    reduction_df.to_excel(writer, sheet_name="reduction_rate_density", index=False)

wb = load_workbook(OUTPUT_FILE)

ws1 = wb["summary"]
autosize_columns(ws1)
center_align(ws1)

ws2 = wb["reduction_rate_density"]
autosize_columns(ws2)
center_align(ws2)

wb.save(OUTPUT_FILE)

print("summary created successfully.")

##################################################################

plt.style.use("seaborn-v0_8-whitegrid")

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10
})

df = reduction_df
models = df["model"].unique()


def plot_reduction_grouped(
    df, model,
    col1, col2,
    label1, label2,
    title, filename,
    sort_by=None
):

    df_model = df[df["model"] == model].copy()
    df_model = df_model[df_model["technique"] != "direct"]

    if df_model.empty:
        return

    if sort_by:
        df_model = df_model.sort_values(sort_by, ascending=False)

    x = np.arange(len(df_model))
    width = 0.35

    y1 = df_model[col1].values
    y2 = df_model[col2].values

    fig, ax = plt.subplots(figsize=(12, 5))

    bars1 = ax.bar(
        x - width/2, y1, width,
        label=label1,
        color="#4C72B0",
        edgecolor="black",
        linewidth=0.6
    )

    bars2 = ax.bar(
        x + width/2, y2, width,
        label=label2,
        color="#DD8452",
        edgecolor="black",
        linewidth=0.6
    )

    ax.set_xticks(x)
    ax.set_xticklabels(df_model["technique"], rotation=35, ha="right")

    ax.set_ylabel("Reduction (%)")
    ax.set_title(f"{title} ({model})")

    ax.axhline(0, color="black", linewidth=0.8)

    ax.legend(frameon=False)

    ax.grid(axis="y", linestyle="--", alpha=0.35)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    def annotate(bars):
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2,
                h + (1 if h >= 0 else -4),
                f"{h:.1f}",
                ha="center",
                va="bottom" if h >= 0 else "top",
                fontsize=9
            )

    annotate(bars1)
    annotate(bars2)

    fig.tight_layout()

    plt.savefig(
        os.path.join(CHARTS_DIR, f"{filename}_{model}.png"),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


for model in models:

    plot_reduction_grouped(
        df,
        model,
        "%▼bandit_rate",
        "%▼semgrep_rate",
        "Bandit Rate Reduction",
        "Semgrep Rate Reduction",
        "Rate Reduction Comparison",
        "reduction_rate",
        sort_by="%▼semgrep_rate"
    )

    plot_reduction_grouped(
        df,
        model,
        "%▼bandit_density",
        "%▼semgrep_density",
        "Bandit Density Reduction",
        "Semgrep Density Reduction",
        "Density Reduction Comparison",
        "reduction_density",
        sort_by="%▼semgrep_density"
    )
print("Plots generated.")

# RANKING PLOTS FOR ALL MODELS

import seaborn as sns

RANK_DIR = os.path.join(BASE_PATH, "charts_ranking")
os.makedirs(RANK_DIR, exist_ok=True)

rank_df = reduction_df.copy()
rank_df = rank_df[rank_df["technique"] != "direct"]

rank_df["rank_rate"] = rank_df.groupby("model")["%▼semgrep_rate"].rank(ascending=False, method="dense")
rank_df["rank_density"] = rank_df.groupby("model")["%▼semgrep_density"].rank(ascending=False, method="dense")

pivot_rate = rank_df.pivot_table(
    index="technique",
    columns="model",
    values="rank_rate",
    aggfunc="mean"
)

pivot_rate = pivot_rate.sort_values(by=list(pivot_rate.columns))

plt.figure(figsize=(10, 7))
sns.heatmap(pivot_rate, annot=True, cmap="Blues_r", fmt=".0f", cbar_kws={"label": "Rank"})
plt.title("Technique Ranking Across Models (Rate Reduction)")
plt.xlabel("Model")
plt.ylabel("Technique")
plt.tight_layout()
plt.savefig(os.path.join(RANK_DIR, "ranking_rate.png"), dpi=300)
plt.close()

pivot_density = rank_df.pivot_table(
    index="technique",
    columns="model",
    values="rank_density",
    aggfunc="mean"
)

pivot_density = pivot_density.sort_values(by=list(pivot_density.columns))

plt.figure(figsize=(10, 7))
sns.heatmap(pivot_density, annot=True, cmap="Greens_r", fmt=".0f", cbar_kws={"label": "Rank"})
plt.title("Technique Ranking Across Models (Density Reduction)")
plt.xlabel("Model")
plt.ylabel("Technique")
plt.tight_layout()
plt.savefig(os.path.join(RANK_DIR, "ranking_density.png"), dpi=300)
plt.close()

# OVERALL RANKING (AVERAGE ACROSS MODELS)

overall_rate = rank_df.groupby("technique")["rank_rate"].mean().sort_values()
overall_density = rank_df.groupby("technique")["rank_density"].mean().sort_values()

plt.figure(figsize=(10, 5))
bars = plt.bar(overall_rate.index, overall_rate.values, color="#4C72B0", edgecolor="black")
plt.xticks(rotation=35, ha="right")
plt.ylabel("Average Rank (Lower = Better)")
plt.title("Overall Technique Ranking (Rate Reduction)")
plt.gca().invert_yaxis()

for bar in bars:
    h = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, h, f"{h:.2f}",
             ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(RANK_DIR, "overall_ranking_rate.png"), dpi=300)
plt.close()


plt.figure(figsize=(10, 5))
bars = plt.bar(overall_density.index, overall_density.values, color="#55A868", edgecolor="black")
plt.xticks(rotation=35, ha="right")
plt.ylabel("Average Rank (Lower = Better)")
plt.title("Overall Technique Ranking (Density Reduction)")
plt.gca().invert_yaxis()

for bar in bars:
    h = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, h, f"{h:.2f}",
             ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(RANK_DIR, "overall_ranking_density.png"), dpi=300)
plt.close()
