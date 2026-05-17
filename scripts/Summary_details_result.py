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
###############
import os
import re
import json
import pandas as pd

from collections import defaultdict


SECURITY_RULE_PATTERNS = [
    ".security.",
    ".injection.",
    "ssrf",
    "xss",
    "xxe",
    "deserialization",
    "eval",
    "exec",
    "subprocess",
    "command-injection",
    "sql-injection",
    "path-traversal",
    "crypto",
    "secrets",
]


TECHNIQUE_SUFFIX_RE = re.compile(r"\(\d+\)$", re.IGNORECASE)


def normalize_technique(name):
    if not name:
        return "unknown"

    name = str(name).strip()
    name = TECHNIQUE_SUFFIX_RE.sub("", name).strip().lower()

    return name


def truncate4(x):
    return round(float(x), 4)


def has_cwe(issue):
    """
    True فقط وقتی CWE واقعاً معتبر باشد.
    """

    if not isinstance(issue, dict):
        return False

    cwe = issue.get("cwe", None)

    if cwe is None:
        return False

    if isinstance(cwe, list):
        for x in cwe:
            if x is None:
                continue

            if str(x).strip():
                return True

        return False

    return str(cwe).strip() != ""


def _issues_from_semgrep_container(container):

    if not isinstance(container, dict):
        return []

    issues = container.get("issues", [])

    return issues if isinstance(issues, list) else []


def is_real_security_issue(issue):
    """
    Detect real security vulnerabilities even when CWE is null.
    """

    if not isinstance(issue, dict):
        return False

    # 1) اگر CWE داشته باشد قطعاً security است
    if has_cwe(issue):
        return True

    # استخراج robust فیلدها
    extra = issue.get("extra", {})

    rule_id = str(
        issue.get(
            "check_id",
            issue.get("rule_id", "")
        )
    ).lower()

    message = str(
        issue.get(
            "message",
            extra.get("message", "")
        )
    ).lower()

    metadata = issue.get(
        "metadata",
        extra.get("metadata", {})
    )

    category = str(
        metadata.get("category", "")
    ).lower()

    text = f"{rule_id} {message}"

    # namespace detection
    for p in SECURITY_RULE_PATTERNS:
        if p in rule_id:
            return True

    # metadata category
    if "security" in category:
        return True

    # fallback text
    for p in SECURITY_RULE_PATTERNS:
        if p in text:
            return True

    return False


def extract_final_semgrep_issues(record):

    if not isinstance(record, dict):
        return []

    iterations = record.get("iterations")

    if isinstance(iterations, list) and len(iterations) > 0:

        last_iteration = iterations[-1]

        if not isinstance(last_iteration, dict):
            return []

        analysis = last_iteration.get("analysis", {})

        if not isinstance(analysis, dict):
            return []

        semgrep_result = analysis.get("semgrep_result", {})

        issues = _issues_from_semgrep_container(semgrep_result)

        return issues

    semgrep_result = record.get("semgrep_result", {})

    issues = _issues_from_semgrep_container(semgrep_result)

    return issues


def analyze_jsonl(filepath):
    """
    Count:
    - real security issues
    - non-security noise

    ONLY from FINAL-CODE semgrep results
    """

    security_count = 0
    noise_count = 0

    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:

            for line in f:

                line = line.strip()

                if not line:
                    continue

                try:
                    record = json.loads(line)

                except Exception:
                    continue

                issues = extract_final_semgrep_issues(record)

                for issue in issues:

                    if is_real_security_issue(issue):
                        security_count += 1
                    else:
                        noise_count += 1

    except Exception as e:
        print(f"ERROR reading {filepath}: {e}")
        return 0, 0

    return security_count, noise_count


def find_matching_jsonl(files, dataset_name):

    dataset_name = str(dataset_name).strip().lower()

    for f in files:

        if not f.lower().endswith(".jsonl"):
            continue

        stem = os.path.splitext(f)[0].strip().lower()

        if stem == dataset_name:
            return f

    return None


# =============================================================
# COLLECT DATA
# =============================================================

security_data = defaultdict(lambda: {
    "security": [],
    "noise": []
})


for root, _, files in os.walk(BASE_DIR):

    rel = os.path.relpath(root, BASE_DIR)

    parts = rel.split(os.sep)

    if len(parts) < 3:
        continue

    dataset = parts[0].strip()
    model = parts[1].strip()

    technique_raw = parts[2].strip()
    technique = normalize_technique(technique_raw)

    matched_jsonl = find_matching_jsonl(files, dataset)

    if not matched_jsonl:
        continue

    jsonl_path = os.path.join(root, matched_jsonl)

    # FIXED TYPO
    security_count, noise_count = analyze_jsonl(jsonl_path)

    key = (dataset, model, technique)

    security_data[key]["security"].append(security_count)
    security_data[key]["noise"].append(noise_count)


# =============================================================
# BUILD DATAFRAME
# =============================================================

cwe_rows = []

for (dataset, model, technique), vals in security_data.items():

    security_runs = vals["security"]
    noise_runs = vals["noise"]

    avg_security = (
        sum(security_runs) / len(security_runs)
        if security_runs else 0
    )

    avg_noise = (
        sum(noise_runs) / len(noise_runs)
        if noise_runs else 0
    )

    cwe_rows.append({
        "dataset": dataset,
        "model": model,
        "technique": technique,
        "avg_security_count": truncate4(avg_security),
        "avg_noise_count": truncate4(avg_noise),
        "runs": len(security_runs)
    })


cwe_df = pd.DataFrame(cwe_rows)


# =============================================================
# SORT
# =============================================================

if not cwe_df.empty:

    cwe_df = (
        cwe_df
        .sort_values(
            by=["dataset", "model", "technique"]
        )
        .reset_index(drop=True)
    )

else:

    cwe_df = pd.DataFrame(columns=[
        "dataset",
        "model",
        "technique",
        "avg_security_count",
        "avg_noise_count",
        "runs"
    ])


# =============================================================
# SAVE TO EXCEL SHEET
# =============================================================

print("\n✅ CWE analysis sheet saved successfully.")


###################

# ... (بخش‌های ابتدایی کد تا رسیدن به ذخیره‌سازی نهایی)

with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
    summary_df.to_excel(writer, sheet_name="summary", index=False)
    reduction_df.to_excel(writer, sheet_name="reduction_rate_density", index=False)
    loc_summary_df.to_excel(writer, sheet_name=LOC_SHEET_NAME, index=False)
    # اضافه کردن شیت تحلیل CWE
    cwe_df.to_excel(writer, sheet_name="cwe_analysis", index=False)

# لود کردن فایل برای اعمال استایل (Auto-size و Center Align)
wb = load_workbook(OUTPUT_FILE)

# اضافه کردن "cwe_analysis" به لیست زیر برای اعمال استایل روی هر 4 شیت
for sheet_name in ["summary", "reduction_rate_density", LOC_SHEET_NAME, "cwe_analysis"]:
    if sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        autosize_columns(ws)
        center_align(ws)

wb.save(OUTPUT_FILE)

print(f"\n✅ All analysis completed. File saved at: {OUTPUT_FILE}")
print("Sheets: summary, reduction_rate_density, Average Generated LOC, cwe_analysis")
