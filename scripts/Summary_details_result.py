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


if os.path.exists(OUTPUT_FILE):
    os.remove(OUTPUT_FILE)

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
        first_val = group[col].dropna().iloc[0]
        decimal_map[col] = count_decimals(first_val)

    means = group[numeric_cols].mean()
    for col in numeric_cols:
        means[col] = round(means[col], decimal_map[col])

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
    ["dataset", "model", "technique",
     "bandit_rate(*100)", "bandit_density(*1000)",
     "semgrep_rate(*100)", "semgrep_density(*1000)",
     "%▼bandit_rate", "%▼bandit_density",
     "%▼semgrep_rate", "%▼semgrep_density"]
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

INPUT_FILE = OUTPUT_FILE


def plot_reduction_grouped(df, model, col1, col2, label1, label2, title, filename):
    df = df[df["model"] == model].copy()
    df = df[df["technique"] != "direct"]
    if df.empty:
        return

    df["avg"] = (df[col1] + df[col2]) / 2
    df = df.sort_values("avg")

    x = np.arange(len(df))
    width = 0.22

    y1 = df[col1].values
    y2 = df[col2].values

    def normalize(vals):
        return (vals - vals.min()) / (vals.max() - vals.min() + 1e-9)

    colors1 = plt.cm.Blues(0.35 + 0.6 * normalize(y1))
    colors2 = plt.cm.Purples(0.35 + 0.6 * normalize(y2))

    fig, ax = plt.subplots(figsize=(11, 5))

    ax.bar(x - width/2, y1, width, label=label1, color=colors1, edgecolor="black", linewidth=0.6)
    ax.bar(x + width/2, y2, width, label=label2, color=colors2, edgecolor="black", linewidth=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(df["technique"], rotation=40, ha="right")

    ax.set_ylabel("Reduction (%)")
    ax.set_title(f"{title} ({model})")
    ax.legend(frameon=False)

    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, f"{filename}_{model}.png"), dpi=300, bbox_inches="tight")
    plt.close()


df = pd.read_excel(INPUT_FILE, sheet_name="reduction_rate_density")
models = df["model"].unique()

for model in models:
    plot_reduction_grouped(df, model,
                           "%▼bandit_rate", "%▼semgrep_rate",
                           "Bandit Rate Reduction", "Semgrep Rate Reduction",
                           "Reduction Rate Comparison", "reduction_rate")

    plot_reduction_grouped(df, model,
                           "%▼bandit_density", "%▼semgrep_density",
                           "Bandit Density Reduction", "Semgrep Density Reduction",
                           "Reduction Density Comparison", "reduction_density")

print("Plots generated.")
