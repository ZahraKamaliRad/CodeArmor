from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from src.core.analyzer import ENABLE_BANDIT, ENABLE_SEMGREP

plt.rcParams["figure.dpi"] = 200
plt.rcParams["font.size"] = 11
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.linewidth"] = 1.2
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["legend.frameon"] = False


TASK_COL = "task_id"
ITER_COL = "iter"
RULE_COL = "rule_id"

ANALYZER_COL = "analyzer"
TASK_BASE_COL = "task_base"


def load_refinement_df(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    for col in (TASK_COL, ITER_COL, RULE_COL):
        if col not in df.columns:
            raise ValueError(
                f"Required column '{col}' not found. Available columns: {list(df.columns)}"
            )

    df[ITER_COL] = pd.to_numeric(df[ITER_COL], errors="coerce")
    df = df.dropna(subset=[ITER_COL])

    df[TASK_COL] = df[TASK_COL].astype(str)

    df[ANALYZER_COL] = df[TASK_COL].apply(
        lambda x: "bandit" if x.endswith("#bandit")
        else "semgrep" if x.endswith("#semgrep")
        else "unknown"
    )

    df[TASK_BASE_COL] = df[TASK_COL].str.replace(r"#(bandit|semgrep)$", "", regex=True)

    def agg_group(g: pd.DataFrame) -> pd.Series:
        mask = g[RULE_COL].notna() & (g[RULE_COL] != "")
        issue_count = int(mask.sum())
        return pd.Series({"issue_count": issue_count})

    grouped = (
        df.groupby([TASK_BASE_COL, ANALYZER_COL, ITER_COL], as_index=False)
        .apply(agg_group,include_groups=False)
        .reset_index(drop=True)
    )

    return grouped


def counts_per_task_iter_df(df: pd.DataFrame) -> pd.DataFrame:
    piv = df.pivot_table(
        index=TASK_BASE_COL,
        columns=ITER_COL,
        values="issue_count",
        aggfunc="sum",
    )

    cols_num = []
    for c in piv.columns:
        try:
            cols_num.append(int(c))
        except Exception:
            pass

    if cols_num:
        full_cols = list(range(min(cols_num), max(cols_num) + 1))
        piv = piv.reindex(columns=full_cols)

    return piv


def totals_point_in_time(pivot: pd.DataFrame) -> pd.Series:
    totals = pivot.fillna(0).sum(axis=0)
    totals.index = totals.index.map(int)
    return totals.sort_index().astype(int)


def totals_carry_forward(pivot: pd.DataFrame) -> pd.Series:
    cf = pivot.ffill(axis=1)
    totals = cf.fillna(0).sum(axis=0)
    totals.index = totals.index.map(int)
    return totals.sort_index().astype(int)


def compute_totals_by_analyzer(
    csv_path: Path, analyzer: str, df: pd.DataFrame | None = None, cumulative: bool = True
) -> pd.Series:

    if df is None:
        df = load_refinement_df(csv_path)
    if df.empty:
        return pd.Series(dtype=int)

    sub = df[df[ANALYZER_COL] == analyzer]
    if sub.empty:
        return pd.Series(dtype=int)

    pivot = counts_per_task_iter_df(sub)
    if pivot.empty:
        return pd.Series(dtype=int)

    return totals_carry_forward(pivot) if cumulative else totals_point_in_time(pivot)


def plot_totals(csv_path: Path,out: Path | None = None,kind: str = "line",show: bool = True,
    df: pd.DataFrame | None = None,verbose: bool = False,cumulative: bool = True,) -> Path | None:

    if not ENABLE_BANDIT and not ENABLE_SEMGREP:
        print("[warn] No analyzer tools enabled (ENABLE_BANDIT/ENABLE_SEMGREP are both False).")
        return None

    if df is None:
        df = load_refinement_df(csv_path)

    bandit_totals = pd.Series(dtype=int)
    semgrep_totals = pd.Series(dtype=int)

    if ENABLE_BANDIT:
        bandit_totals = compute_totals_by_analyzer(
            csv_path, "bandit", df=df, cumulative=cumulative
        )

    if ENABLE_SEMGREP:
        semgrep_totals = compute_totals_by_analyzer(
            csv_path, "semgrep", df=df, cumulative=cumulative
        )

    if bandit_totals.empty and semgrep_totals.empty:
        print("[warn] No data to plot.")
        return None

    plt.figure(figsize=(8, 4.5))

    if kind == "bar":
        kind = "line"

    if ENABLE_BANDIT and not bandit_totals.empty:
        plt.plot(
            list(bandit_totals.index),
            list(bandit_totals.values),
            linewidth=2.2,
            marker="o",
            markersize=5,
            label="Bandit",
        )

    if ENABLE_SEMGREP and not semgrep_totals.empty:
        plt.plot(
            list(semgrep_totals.index),
            list(semgrep_totals.values),
            linewidth=2.2,
            marker="s",
            markersize=5,
            label="Semgrep",
        )

    all_x = sorted(set(bandit_totals.index).union(set(semgrep_totals.index)))
    if all_x:
        xmin, xmax = int(min(all_x)), int(max(all_x))
        plt.xlim(xmin, xmax)
        plt.xticks(range(xmin, xmax + 1, 1))

    y_all = list(bandit_totals.values) + list(semgrep_totals.values)
    ymax = max(y_all) if y_all else 0
    step = 5
    ymax_rounded = ((int(ymax) + step - 1) // step) * step
    plt.yticks(range(0, ymax_rounded + step, step))

    plt.xlabel("Iterations")
    plt.ylabel("Total vulnerabilities")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()

    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()
        return out
    else:
        if show:
            plt.show()
        else:
            plt.close()
        return None
