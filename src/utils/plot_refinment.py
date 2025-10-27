
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import os, sys, subprocess

TASK_COL = "task_id"
ITER_COL = "iter"
RULE_COL = "rule_id"

def load_refinement_df(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    for col in (TASK_COL, ITER_COL, RULE_COL):
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found. Available columns: {list(df.columns)}")
    df[ITER_COL] = pd.to_numeric(df[ITER_COL], errors="coerce")
    df = df[df[RULE_COL].notna() & (df[RULE_COL] != "")]
    return df

def counts_per_task_iter_df(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby([TASK_COL, ITER_COL])[RULE_COL]
          .nunique()
          .unstack(fill_value=0)
    )

def compute_totals(csv_path: Path, df: pd.DataFrame | None = None) -> pd.Series:
    if df is None:
        df = load_refinement_df(csv_path)
    if df.empty:
        return pd.Series(dtype=int)
    pivot = counts_per_task_iter_df(df)
    totals = pivot.sum(axis=0)
    totals.index = totals.index.map(lambda x: int(x) if pd.notna(x) else x)
    return totals.sort_index().astype(int)

def plot_totals(csv_path: Path, out: Path | None = None, kind: str = "line", show: bool = True, df: pd.DataFrame | None = None, verbose: bool = False) -> Path | None:
    totals = compute_totals(csv_path, df=df)
    if verbose:
        print("\n=== Total Vulnerabilities by Iteration ===")
        for it, val in totals.items():
            print(f"iter {it}: {val}")
    plt.figure(figsize=(8, 4.5))
    x = list(totals.index)
    y = list(totals.values)
    if kind == "bar":
        plt.bar(x, y)
    else:
        plt.plot(x, y, marker="o")
    plt.title("Total Vulnerabilities vs Iterations")
    plt.xlabel("Iterations")
    plt.xticks(x)
    plt.ylabel("Total vulnerabilities")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=200)
        if show:
            try:
                os.startfile(str(out))
            except AttributeError:
                try:
                    if sys.platform == "darwin":
                        subprocess.run(["open", str(out)], check=False)
                    else:
                        subprocess.run(["xdg-open", str(out)], check=False)
                except Exception:
                    pass
            except Exception:
                pass
        plt.close()
        return out
    else:
        if show:
            plt.show()
        else:
            plt.close()
        return None

def label_change(start: int, end: int) -> str:
    if start > 0 and end == 0: return "zero"
    if start > 0 and end > 0 and end < start: return "decreased"
    if start > 0 and end > 0 and end > start: return "increased"
    if start == 0 and end > 0: return "introduced"
    if start > 0 and end == start: return "unchanged"
    return "none"

def refinment_summary(csv_path: Path, metrics_txt: Path | None = None, df: pd.DataFrame | None = None, write: bool = True) -> dict:
    if df is None:
        if not Path(csv_path).exists():
            text = "\n=== Refinement Summary (0 -> 0) ===\nzero --> 0\ndecreased --> 0\nincreased --> 0\nintroduced --> 0\nunchanged --> 0\n"
            if write and metrics_txt:
                metrics_txt.parent.mkdir(parents=True, exist_ok=True)
                with open(metrics_txt, "a", encoding="utf-8") as f:
                    f.write(text)
            return {"end_iter": 0, "buckets": {"zero":0,"decreased":0,"increased":0,"introduced":0,"unchanged":0}, "text": text}
        df = load_refinement_df(csv_path)
    if df.empty:
        text = "\n=== Refinement Summary (0 -> 0) ===\nzero --> 0\ndecreased --> 0\nincreased --> 0\nintroduced --> 0\nunchanged --> 0\n"
        if write and metrics_txt:
            metrics_txt.parent.mkdir(parents=True, exist_ok=True)
            with open(metrics_txt, "a", encoding="utf-8") as f:
                f.write(text)
        return {"end_iter": 0, "buckets": {"zero":0,"decreased":0,"increased":0,"introduced":0,"unchanged":0}, "text": text}
    pivot_all = counts_per_task_iter_df(df)
    iters_present = []
    for c in pivot_all.columns:
        try:
            iters_present.append(int(c))
        except Exception:
            pass
    if 0 not in iters_present:
        raise ValueError("Expected baseline iter 0 is missing.")
    end_iter = max(iters_present)
    comp = pd.DataFrame(index=pivot_all.index)
    comp["start_iter"] = pivot_all.get(0, 0)
    comp["end_iter"] = pivot_all.get(end_iter, 0)
    comp = comp.fillna(0).astype(int)
    cats = comp.apply(lambda r: label_change(int(r["start_iter"]), int(r["end_iter"])), axis=1)
    order = ["zero", "decreased", "increased", "introduced", "unchanged"]
    vc = cats.value_counts().reindex(order, fill_value=0)
    text_lines = [
        "",
        f"=== Refinement Summary (0 -> {end_iter}) ===",
        f"zero --> {int(vc.get('zero', 0))}",
        f"decreased --> {int(vc.get('decreased', 0))}",
        f"increased --> {int(vc.get('increased', 0))}",
        f"introduced --> {int(vc.get('introduced', 0))}",
        f"unchanged --> {int(vc.get('unchanged', 0))}",
        ""
    ]
    text = "\n".join(text_lines)
    if write and metrics_txt:
        metrics_txt.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_txt, "a", encoding="utf-8") as f:
            f.write(text)
    return {
        "end_iter": int(end_iter),
        "buckets": {k: int(vc.get(k, 0)) for k in order},
        "text": text
    }
