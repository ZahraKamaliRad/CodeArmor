import os
import re
import pandas as pd
from decimal import Decimal, ROUND_DOWN
from openpyxl import load_workbook
from openpyxl.styles import Alignment
from openpyxl.utils import get_column_letter

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_PATH)
BASE_DIR = os.path.join(PROJECT_ROOT, "outputs")
OUTPUT_FILE = os.path.join(BASE_PATH, "summary.xlsx")

TECHNIQUE_SUFFIX_RE = re.compile(r"\(\d+\)$")
LOC_RE = re.compile(
    r"^\s*total\s+LOC\s+\(L\)\s*:\s*(\d+)\s*$",
    re.IGNORECASE | re.MULTILINE
)

def autosize_columns(ws):
    for column in ws.columns:
        max_length = 0
        col = column[0].column

        for cell in column:
            if cell.value is not None:
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
    except Exception:
        return 0


def truncate4(x):
    try:
        return float(
            Decimal(str(x)).quantize(
                Decimal("0.0001"),
                rounding=ROUND_DOWN
            )
        )
    except Exception:
        return x


def safe_sheet_name(name):
    invalid = ['\\', '/', '*', '[', ']', ':', '?']

    for ch in invalid:
        name = name.replace(ch, "_")

    return name[:31]

try:
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)
except PermissionError:
    pass

dataframes = []

for root, _, files in os.walk(BASE_DIR):
    for file in files:

        if not file.endswith(".xlsx"):
            continue

        if file.startswith("~$"):
            continue

        file_path = os.path.join(root, file)

        try:
            df = pd.read_excel(file_path)
            dataframes.append(df)
        except Exception:
            continue

if not dataframes:
    raise ValueError("No valid Excel files found.")

all_data = pd.concat(dataframes, ignore_index=True)

all_data["technique"] = (
    all_data["technique"]
    .astype(str)
    .str.replace(TECHNIQUE_SUFFIX_RE, "", regex=True)
    .str.strip())

group_cols = ["dataset", "model", "technique"]
summary_rows = []

for _, group in all_data.groupby(group_cols):

    numeric_cols = group.select_dtypes(include="number").columns

    decimal_map = {}

    for col in numeric_cols:
        values = group[col].dropna()

        if len(values) == 0:
            decimal_map[col] = 2
        else:
            decimal_map[col] = count_decimals(values.iloc[0])

    means = group[numeric_cols].mean()

    for col in numeric_cols:
        means[col] = round(means[col], decimal_map.get(col, 2))

    base_row = group.iloc[0].copy()

    for col in numeric_cols:
        base_row[col] = means[col]

    summary_rows.append(base_row)

summary_df = pd.DataFrame(summary_rows)
summary_df = summary_df[all_data.columns]

summary_df = (
    summary_df
    .sort_values(by=["dataset", "model", "technique"])
    .reset_index(drop=True)
)

numeric_cols = summary_df.select_dtypes(include="number").columns

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

            "%▼bandit_rate": truncate4(
                safe_reduction(d_bandit_rate,row.get("bandit_rate", 0))
            ),

            "%▼bandit_density": truncate4(
                safe_reduction(d_bandit_density,row.get("bandit_density", 0))
            ),

            "%▼semgrep_rate": truncate4(
                safe_reduction(d_semgrep_rate,row.get("semgrep_rate", 0))
            ),

            "%▼semgrep_density": truncate4(
                safe_reduction(d_semgrep_density,row.get("semgrep_density", 0))
            )
        })

reduction_df = pd.DataFrame(reduction_rows)

reduction_df = reduction_df[
    [
        "dataset",
        "model",
        "technique",

        "bandit_rate",
        "bandit_density",
        "semgrep_rate",
        "semgrep_density",

        "%▼bandit_rate",
        "%▼bandit_density",
        "%▼semgrep_rate",
        "%▼semgrep_density",
    ]
]

reduction_df = (reduction_df
    .sort_values(by=["dataset", "model", "technique"])
    .reset_index(drop=True))

loc_rows = []

for root, _, files in os.walk(BASE_DIR):
    for file in files:

        if not file.endswith(".txt"):
            continue

        full_path = os.path.join(root, file)
        rel_path = os.path.relpath(full_path, BASE_DIR)
        parts = rel_path.split(os.sep)

        if len(parts) < 4:
            continue

        dataset = parts[0]
        model = parts[1]

        technique_raw = parts[2]
        technique = (TECHNIQUE_SUFFIX_RE.sub("", technique_raw).strip())
        try:
            with open(full_path,"r",encoding="utf-8",errors="ignore") as f:
                content = f.read()

        except Exception:
            continue

        match = LOC_RE.search(content)

        if not match:
            continue

        loc_rows.append({"dataset": dataset,"model": model,"technique": technique,
                         "total LOC (L)": int(match.group(1))})

if loc_rows:

    loc_df = pd.DataFrame(loc_rows)

    loc_summary_df = (loc_df.groupby(["dataset", "model", "technique"],dropna=False)["total LOC (L)"]
        .mean()
        .reset_index()
        .sort_values(by=["dataset", "model", "technique"])
        .reset_index(drop=True)
    )

    loc_summary_df["total LOC (L)"] = (loc_summary_df["total LOC (L)"].apply(truncate4))

else:
    loc_summary_df = pd.DataFrame(
        columns=["dataset","model","technique","total LOC (L)"])

LOC_SHEET_NAME = safe_sheet_name("Average Generated LOC")

with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:

    summary_df.to_excel(writer,sheet_name="ُSummary",index=False)

    reduction_df.to_excel(writer,sheet_name="Reduction_rate_density",index=False)

    loc_summary_df.to_excel(writer,sheet_name=LOC_SHEET_NAME,index=False)

wb = load_workbook(OUTPUT_FILE)

for sheet_name in ["summary","reduction_rate_density",LOC_SHEET_NAME]:
    ws = wb[sheet_name]
    autosize_columns(ws)
    center_align(ws)

wb.save(OUTPUT_FILE)

print("summary.xlsx created successfully.")
