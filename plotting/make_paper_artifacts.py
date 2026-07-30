import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Iterable

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from cove_paths import default_plotting_out_dir

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_CONFIGS = ["CSV+4090", "CSV+5090", "H20"]


def _read_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_results(results_dir: str) -> pd.DataFrame:
    files: list[str] = []
    for root, _dirs, names in os.walk(results_dir):
        for name in names:
            if name.endswith(".jsonl"):
                files.append(os.path.join(root, name))
    rows: list[dict[str, Any]] = []
    for p in sorted(files):
        try:
            rows.extend(_read_jsonl(p))
        except Exception:
            continue
    if not rows:
        return pd.DataFrame()

    df = pd.json_normalize(rows)
    # Normalize columns we need.
    for c in ["input_tokens", "output_tokens", "batch"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    for c in ["ttft_ms", "total_ms", "tokens_per_s", "transfer_ms"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Derive run_id so multiple files can be aggregated with error bars.
    if "started_at" in df.columns:
        df["run_id"] = df["started_at"].astype(str)
    else:
        df["run_id"] = "unknown"
    return df


@dataclass(frozen=True)
class CurveSpec:
    name: str
    color: str
    marker: str


CURVES = [
    CurveSpec("CSV+4090", "#1f77b4", "o"),
    CurveSpec("CSV+5090", "#ff7f0e", "s"),
    CurveSpec("H20", "#2ca02c", "^"),
]


def _style_axes(ax):
    ax.grid(True, which="major", alpha=0.25)
    ax.set_axisbelow(True)


def _agg(df: pd.DataFrame, *, metric: str, group_cols: list[str]) -> pd.DataFrame:
    """
    Aggregate across runs to mean±std. If only one run exists, std will be NaN -> plotted without bars.
    """
    g = df.groupby(group_cols, dropna=False)[metric]
    out = g.agg(["mean", "std", "count"]).reset_index()
    out = out.rename(columns={"mean": metric, "std": f"{metric}_std", "count": "n"})
    return out


def _plot_line_or_errorbar(ax, *, x: np.ndarray, y: np.ndarray, ystd: np.ndarray, n: np.ndarray, label: str, color: str, marker: str):
    # Only draw error bars when we have repeated runs (n>1) and finite std.
    has_err = (n > 1) & np.isfinite(ystd)
    if bool(np.any(has_err)):
        ax.errorbar(x, y, yerr=np.where(has_err, ystd, np.nan), label=label, color=color, marker=marker, linewidth=2, capsize=3)
    else:
        ax.plot(x, y, label=label, color=color, marker=marker, linewidth=2)


def fig_ttft_vs_input(df: pd.DataFrame, out_path: str, *, batch: int = 1):
    # Line plot: TTFT(ms) vs input_tokens, for a fixed batch.
    sub = df[(df["batch"] == batch) & (df["config"].isin([c.name for c in CURVES]))].copy()
    if sub.empty:
        return
    agg = _agg(sub, metric="ttft_ms", group_cols=["config", "input_tokens"])

    plt.figure(figsize=(7.2, 4.2), dpi=200)
    ax = plt.gca()
    for c in CURVES:
        d = agg[agg["config"] == c.name].sort_values("input_tokens")
        if d.empty:
            continue
        x = d["input_tokens"].astype(float).to_numpy()
        y = d["ttft_ms"].to_numpy()
        ystd = d["ttft_ms_std"].to_numpy()
        n = d["n"].to_numpy()
        _plot_line_or_errorbar(ax, x=x, y=y, ystd=ystd, n=n, label=c.name, color=c.color, marker=c.marker)

    ax.set_xlabel("Input length (tokens)")
    ax.set_ylabel("TTFT (ms)")
    ax.set_title(f"TTFT vs input length (batch={batch})")
    _style_axes(ax)
    ax.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def fig_throughput_vs_batch(df: pd.DataFrame, out_path: str, *, input_tokens: int = 512):
    # Grouped bar chart: throughput (tokens/s) vs batch for a fixed input_tokens.
    sub = df[(df["input_tokens"] == input_tokens) & (df["config"].isin([c.name for c in CURVES]))].copy()
    if sub.empty:
        return
    agg = _agg(sub, metric="tokens_per_s", group_cols=["config", "batch"])

    plt.figure(figsize=(7.2, 4.2), dpi=200)
    ax = plt.gca()
    batches = sorted(int(x) for x in agg["batch"].dropna().unique().tolist())
    if not batches:
        return
    x0 = np.arange(len(batches), dtype=float)
    width = 0.8 / max(1, len(CURVES))
    for i, c in enumerate(CURVES):
        d = agg[agg["config"] == c.name].set_index("batch")
        y = np.array([float(d.loc[b, "tokens_per_s"]) if b in d.index else np.nan for b in batches], dtype=float)
        ystd = np.array([float(d.loc[b, "tokens_per_s_std"]) if b in d.index else np.nan for b in batches], dtype=float)
        n = np.array([int(d.loc[b, "n"]) if b in d.index else 0 for b in batches], dtype=int)
        xpos = x0 + (i - (len(CURVES) - 1) / 2) * width
        ax.bar(xpos, y, width=width, label=c.name, color=c.color, alpha=0.9, edgecolor="white", linewidth=0.6)
        # Error bars only when repeated runs exist.
        has_err = (n > 1) & np.isfinite(ystd)
        if bool(np.any(has_err)):
            ax.errorbar(xpos[has_err], y[has_err], yerr=ystd[has_err], fmt="none", ecolor="#111827", elinewidth=1.2, capsize=3, alpha=0.9)

    ax.set_xlabel("Batch size")
    ax.set_ylabel("Throughput (tokens/s, approx)")
    ax.set_title(f"Throughput vs batch (input={input_tokens} tokens)")
    _style_axes(ax)
    ax.set_xticks(x0)
    ax.set_xticklabels([str(b) for b in batches])
    ax.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def fig_total_ms_vs_input(df: pd.DataFrame, out_path: str, *, batch: int = 1):
    # Box plot: distribution of total_ms across ALL scenarios for each config.
    # This gives a different view than a curve and highlights variability/outliers.
    sub = df[df["config"].isin([c.name for c in CURVES])].copy()
    if sub.empty:
        return

    plt.figure(figsize=(7.2, 4.2), dpi=200)
    ax = plt.gca()

    data = []
    labels = []
    colors = []
    for c in CURVES:
        v = sub[sub["config"] == c.name]["total_ms"].dropna().to_numpy(dtype=float)
        if v.size == 0:
            continue
        data.append(v)
        labels.append(c.name)
        colors.append(c.color)

    bp = ax.boxplot(
        data,
        labels=labels,
        patch_artist=True,
        showfliers=True,
        medianprops=dict(color="#111827", linewidth=1.6),
        boxprops=dict(linewidth=1.0),
        whiskerprops=dict(linewidth=1.0),
        capprops=dict(linewidth=1.0),
    )
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.28)
        patch.set_edgecolor(col)

    ax.set_ylabel("Total time (ms)")
    ax.set_title("Total time distribution across scenarios (all inputs/batches)")
    _style_axes(ax)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def latex_summary_table(df: pd.DataFrame, out_path: str):
    """
    A compact summary at one operating point: input=512, batch=1.
    """
    sub = df[(df["input_tokens"] == 512) & (df["batch"] == 1) & (df["config"].isin([c.name for c in CURVES]))].copy()
    if sub.empty:
        return
    agg = sub.groupby("config", dropna=False).agg(
        ttft_ms_mean=("ttft_ms", "mean"),
        ttft_ms_std=("ttft_ms", "std"),
        total_ms_mean=("total_ms", "mean"),
        total_ms_std=("total_ms", "std"),
        tps_mean=("tokens_per_s", "mean"),
        tps_std=("tokens_per_s", "std"),
        n=("ttft_ms", "count"),
    )
    agg = agg.reset_index()

    def pm(m, s):
        if pd.isna(s) or float(agg.loc[0, "n"]) <= 1:
            return f"{m:.2f}"
        return f"{m:.2f} $\\pm$ {s:.2f}"

    rows = []
    for _, r in agg.iterrows():
        rows.append(
            {
                "Config": r["config"],
                "TTFT (ms)": pm(r["ttft_ms_mean"], r["ttft_ms_std"]),
                "Total (ms)": pm(r["total_ms_mean"], r["total_ms_std"]),
                "Throughput (tok/s)": pm(r["tps_mean"], r["tps_std"]),
                "N": int(r["n"]),
            }
        )
    table_df = pd.DataFrame(rows)

    tex = table_df.to_latex(
        index=False,
        escape=False,
        column_format="lrrrr",
        caption="Latency and throughput comparison at input=512 tokens, batch=1 (mean$\\pm$std across runs when available).",
        label="tab:ourscheme_eval_summary",
    )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(tex)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="eval-results/llama-2-7b-hf")
    ap.add_argument("--out-dir", default=None, help="Default: plotting/<model-slug>/ inferred from --results-dir")
    ap.add_argument("--model-ref", default=None, help="Optional model path to pick plotting output subdir")
    ap.add_argument("--configs", default=",".join(DEFAULT_CONFIGS), help="Comma-separated configs to plot")
    ap.add_argument("--batch-for-curves", type=int, default=1)
    ap.add_argument("--input-for-throughput", type=int, default=512)
    args = ap.parse_args()

    out_dir = args.out_dir or default_plotting_out_dir(args.results_dir, args.model_ref)
    os.makedirs(out_dir, exist_ok=True)
    df = load_results(args.results_dir)
    if df.empty:
        raise SystemExit(f"No rows loaded from {args.results_dir}")

    wanted = [x.strip() for x in str(args.configs).split(",") if x.strip()]
    global CURVES
    palette = {"CSV+4090": "#1f77b4", "CSV+5090": "#ff7f0e", "H20": "#2ca02c"}
    markers = {"CSV+4090": "o", "CSV+5090": "s", "H20": "^"}
    CURVES = [CurveSpec(n, palette.get(n, "#444444"), markers.get(n, "o")) for n in wanted]

    fig_ttft_vs_input(df, os.path.join(out_dir, "fig_ttft_vs_input.png"), batch=args.batch_for_curves)
    fig_total_ms_vs_input(df, os.path.join(out_dir, "fig_total_ms_vs_input.png"), batch=args.batch_for_curves)
    fig_throughput_vs_batch(df, os.path.join(out_dir, "fig_throughput_vs_batch.png"), input_tokens=args.input_for_throughput)
    latex_summary_table(df, os.path.join(out_dir, "table_summary.tex"))

    print(f"Wrote artifacts to: {out_dir}")


if __name__ == "__main__":
    main()

