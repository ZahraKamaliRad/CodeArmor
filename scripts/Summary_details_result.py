import os
import pandas as pd
from decimal import Decimal
from openpyxl import load_workbook
from openpyxl.utils import get_column_letter
from openpyxl.styles import Alignment
import matplotlib.pyplot as plt
import numpy as np
from decimal import Decimal, ROUND_DOWN


BASE_PATH = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_PATH)
BASE_DIR = os.path.join(PROJECT_ROOT, "outputs")
OUTPUT_FILE = os.path.join(BASE_PATH, "summary.xlsx")

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

def truncate4(x):
    try:
        return float(Decimal(x).quantize(Decimal('0.0001'), rounding=ROUND_DOWN))
    except:
        return x

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

numeric_cols = summary_df.select_dtypes(include='number').columns
for col in numeric_cols:
    summary_df[col] = summary_df[col].apply(truncate4)

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
            "bandit_rate": truncate4(row.get("bandit_rate", 0)),
            "bandit_density": truncate4(row.get("bandit_density", 0)),
            "semgrep_rate": truncate4(row.get("semgrep_rate", 0)),
            "semgrep_density": truncate4(row.get("semgrep_density", 0)),
            "%▼bandit_rate": truncate4(safe_reduction(d_bandit_rate, row.get("bandit_rate", 0))),
            "%▼bandit_density": truncate4(safe_reduction(d_bandit_density, row.get("bandit_density", 0))),
            "%▼semgrep_rate": truncate4(safe_reduction(d_semgrep_rate, row.get("semgrep_rate", 0))),
            "%▼semgrep_density": truncate4(safe_reduction(d_semgrep_density, row.get("semgrep_density", 0))),
        })

reduction_df = pd.DataFrame(reduction_rows)

reduction_df = reduction_df[
    [
        "dataset", "model", "technique",
        "bandit_rate", "bandit_density",
        "semgrep_rate", "semgrep_density",
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

# python scripts\summary_details_result.py