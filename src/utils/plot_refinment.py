from pathlib import Path
import json
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

ANALYZER_COL = "analyzer"
TASK_BASE_COL = "task_base"


def load_refinement_df(jsonl_path: Path) -> pd.DataFrame:

    rows = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:

            if not line.strip():
                continue

            rec = json.loads(line)

            task = rec.get("task_id") or rec.get("id") or rec.get("task")

            init = rec.get("initial_analysis", {})

            if ENABLE_BANDIT:
                b = len(init.get("bandit_result", {}).get("issues", []))
                rows.append(
                    {
                        TASK_BASE_COL: task,
                        ANALYZER_COL: "bandit",
                        ITER_COL: 0,
                        "issue_count": b,
                    }
                )

            if ENABLE_SEMGREP:
                s = len(init.get("semgrep_result", {}).get("issues", []))
                rows.append(
                    {
                        TASK_BASE_COL: task,
                        ANALYZER_COL: "semgrep",
                        ITER_COL: 0,
                        "issue_count": s,
                    }
                )

            for it in rec.get("iterations", []):

                i = it.get("iter")

                analysis = it.get("analysis", {})

                if ENABLE_BANDIT:
                    b = len(analysis.get("bandit_result", {}).get("issues", []))
                    rows.append(
                        {
                            TASK_BASE_COL: task,
                            ANALYZER_COL: "bandit",
                            ITER_COL: i,
                            "issue_count": b,
                        }
                    )

                if ENABLE_SEMGREP:
                    s = len(analysis.get("semgrep_result", {}).get("issues", []))
                    rows.append(
                        {
                            TASK_BASE_COL: task,
                            ANALYZER_COL: "semgrep",
                            ITER_COL: i,
                            "issue_count": s,
                        }
                    )

    df = pd.DataFrame(rows)

    df[ITER_COL] = pd.to_numeric(df[ITER_COL], errors="coerce")
    df = df.dropna(subset=[ITER_COL])

    return df


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


def compute_totals_by_analyzer(jsonl_path: Path,analyzer: str,df: pd.DataFrame | None = None,
    cumulative: bool = True) -> pd.Series:

    if df is None:
        df = load_refinement_df(jsonl_path)

    if df.empty:
        return pd.Series(dtype=int)

    sub = df[df[ANALYZER_COL] == analyzer]

    if sub.empty:
        return pd.Series(dtype=int)

    pivot = counts_per_task_iter_df(sub)

    if pivot.empty:
        return pd.Series(dtype=int)

    return totals_carry_forward(pivot) if cumulative else totals_point_in_time(pivot)


def plot_totals(jsonl_path: Path,out: Path | None = None,kind: str = "line",show: bool = True,
    df: pd.DataFrame | None = None,verbose: bool = False,cumulative: bool = True) -> Path | None:

    if not ENABLE_BANDIT and not ENABLE_SEMGREP:

        print("[warn] No analyzer tools enabled.")
        return None

    if df is None:
        df = load_refinement_df(jsonl_path)

    bandit_totals = pd.Series(dtype=int)
    semgrep_totals = pd.Series(dtype=int)

    print("\n--- DEBUG PLOT DATA ---")

    if ENABLE_BANDIT:

        bandit_totals = compute_totals_by_analyzer(
            jsonl_path, "bandit", df=df, cumulative=cumulative
        )

        print("Bandit totals per iteration:")
        print(bandit_totals)

    if ENABLE_SEMGREP:

        semgrep_totals = compute_totals_by_analyzer(
            jsonl_path, "semgrep", df=df, cumulative=cumulative
        )

        print("Semgrep totals per iteration:")
        print(semgrep_totals)

    print("-----------------------\n")

    if bandit_totals.empty and semgrep_totals.empty:
        print("[warn] No data to plot.")
        return None
    all_iters = set(bandit_totals.index).union(set(semgrep_totals.index))

    has_refinement = any(i > 0 for i in all_iters)

    if not has_refinement:
        if verbose:
            print("[info] Only iteration 0 present — skipping plot.")
        return None


    plt.figure(figsize=(8, 4.5))

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
