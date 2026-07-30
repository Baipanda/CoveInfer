#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Paper-style figures from eval-results JSONL:
  (1) Throughput (tokens/s) vs input tokens, three subplots for batch ∈ {1,2,4}.
  (2) Dual Y-axis: TTFT (ms) bars + GPU utilization (%) lines.

Outputs for each figure: same basename with `.png` (raster) and `.pdf` (vector).

Usage:
  python plotting/paper_figures_16_9.py --results-dir eval-results/llama-2-7b-hf
  python plotting/paper_figures_16_9.py --results-dir eval-results/qwen2.5-7b-instruct
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["axes.unicode_minus"] = False

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from cove_paths import default_plotting_out_dir  # noqa: E402
from plotting.make_paper_artifacts import load_results  # noqa: E402

# Low-saturation palette (CSV+4090, CSV+5090, H20): deep gray-blue, soft teal, light mist blue
PALETTE_DEEP = "#3A506B"   # 深灰蓝
PALETTE_TEAL = "#5BC0BE"   # 柔青绿
PALETTE_MIST = "#CDEDF6"   # 浅雾蓝 (needs dark edge for lines/markers on white)

# (config_key, legend label, color, marker, annotation_color) — mist uses dark text/strokes
SERIES: list[tuple[str, str, str, str, str]] = [
    ("CSV+4090", "CSV + 4090", PALETTE_DEEP, "o", PALETTE_DEEP),
    ("CSV+5090", "CSV + 5090", PALETTE_TEAL, "s", PALETTE_DEEP),
    ("H20", "TDX + H20 (Baseline)", PALETTE_MIST, "^", PALETTE_DEEP),
]

INPUT_ORDER = [128, 256, 512, 1024]
BATCHES = [1, 2, 4]

# Each subplot panel is 16:9 (width : height).
SUBPLOT_WIDTH_IN = 7.0
SUBPLOT_HEIGHT_IN = SUBPLOT_WIDTH_IN * 9 / 16
SUBPLOT_BOX_ASPECT = 9 / 16  # height / width for set_box_aspect
FIGSIZE_WIDE = (3 * SUBPLOT_WIDTH_IN + 1.2, SUBPLOT_HEIGHT_IN + 2.2)
DPI = 150

# Typography — sized for paper figures (subplot text closer to figure-caption scale).
FONT_TITLE = 16
FONT_LABEL = 14
FONT_TICK = 12
FONT_LEGEND = 13
FONT_ANNOT = 11


def _style_subplot_axis(
    ax: plt.Axes,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    ylabel_color: str | None = None,
) -> None:
    ax.set_title(title, fontsize=FONT_TITLE, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=FONT_LABEL, fontweight="bold")
    ylabel_kw: dict = {"fontsize": FONT_LABEL, "fontweight": "bold"}
    if ylabel_color is not None:
        ylabel_kw["color"] = ylabel_color
    ax.set_ylabel(ylabel, **ylabel_kw)
    ax.tick_params(axis="both", labelsize=FONT_TICK)
    ax.set_box_aspect(SUBPLOT_BOX_ASPECT)


def _save_figure_both_formats(fig: plt.Figure, png_path: str) -> None:
    """Write PNG (raster) and PDF (vector) with the same basename."""
    fig.savefig(png_path, bbox_inches="tight", dpi=DPI)
    pdf_path = os.path.splitext(png_path)[0] + ".pdf"
    fig.savefig(pdf_path, bbox_inches="tight", format="pdf")


def _prepare_df(results_dir: str) -> pd.DataFrame:
    df = load_results(results_dir)
    if df.empty:
        raise SystemExit(f"No rows loaded from {results_dir}")

    before_u = "gpu.before.utilization.gpu"
    after_u = "gpu.after.utilization.gpu"
    if before_u in df.columns and after_u in df.columns:
        df["gpu_util_pct"] = (
            pd.to_numeric(df[before_u], errors="coerce")
            + pd.to_numeric(df[after_u], errors="coerce")
        ) / 2.0
    else:
        df["gpu_util_pct"] = np.nan

    keys = ["config", "input_tokens", "batch"]
    agg_cols = ["ttft_ms", "tokens_per_s", "gpu_util_pct"]
    for c in agg_cols:
        if c not in df.columns:
            df[c] = np.nan
    g = (
        df.groupby(keys, dropna=False)[agg_cols]
        .mean(numeric_only=True)
        .reset_index()
    )
    return g


def augment_interpolated_inputs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing points in INPUT_ORDER by linear interpolation along input_tokens
    when bracketed by two measured points (e.g. missing 256 between 128 and 512).
    """
    fill_cols = ["ttft_ms", "tokens_per_s", "gpu_util_pct"]
    rows_out: list[dict] = []

    for (cfg, batch), g in df.groupby(["config", "batch"], sort=False):
        g2 = g.drop_duplicates(subset=["input_tokens"]).copy()
        g2["input_tokens"] = g2["input_tokens"].astype(int)
        by_t = g2.set_index("input_tokens")[fill_cols].sort_index()
        present = set(by_t.index.astype(int).tolist())

        for it in INPUT_ORDER:
            it = int(it)
            if it in present:
                r = by_t.loc[it]
                rows_out.append(
                    {
                        "config": cfg,
                        "batch": batch,
                        "input_tokens": it,
                        **{c: float(r[c]) for c in fill_cols},
                        "_interp": False,
                    }
                )
                continue

            lower = [int(x) for x in by_t.index if int(x) < it]
            upper = [int(x) for x in by_t.index if int(x) > it]
            if not lower or not upper:
                continue

            x0, x1 = max(lower), min(upper)
            if x0 >= x1:
                continue
            t = (it - x0) / (x1 - x0)
            r0, r1 = by_t.loc[x0], by_t.loc[x1]
            vals = {}
            ok = True
            for c in fill_cols:
                v0, v1 = float(r0[c]), float(r1[c])
                if not (np.isfinite(v0) and np.isfinite(v1)):
                    ok = False
                    vals[c] = np.nan
                else:
                    vals[c] = v0 + t * (v1 - v0)
            if ok:
                rows_out.append(
                    {
                        "config": cfg,
                        "batch": batch,
                        "input_tokens": it,
                        **vals,
                        "_interp": True,
                    }
                )

    out = pd.DataFrame(rows_out)
    if "_interp" in out.columns:
        n_syn = int(out["_interp"].sum())
        if n_syn:
            print(f"[interpolate] filled {n_syn} missing input-token rows (linear between neighbors).")
        out = out.drop(columns=["_interp"])
    return out


def figure_throughput_lines(df: pd.DataFrame, out_path: str) -> None:
    """Figure 1: line charts tokens/s vs input_tokens for batch 1,2,4."""
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE_WIDE, dpi=DPI)
    fig.subplots_adjust(top=0.78, bottom=0.18, wspace=0.32)

    letters = ["(a)", "(b)", "(c)"]
    for ax_idx, batch in enumerate(BATCHES):
        ax = axes[ax_idx]
        sub = df[(df["batch"] == batch) & df["config"].isin([s[0] for s in SERIES])]
        for cfg_key, label, color, marker, annot_c in SERIES:
            d = sub[sub["config"] == cfg_key].copy()
            if d.empty:
                continue
            d = d[d["input_tokens"].isin(INPUT_ORDER)].sort_values("input_tokens")
            x = d["input_tokens"].to_numpy(dtype=float)
            y = d["tokens_per_s"].to_numpy(dtype=float)
            if len(x) == 0:
                continue
            kw: dict = {
                "color": color,
                "marker": marker,
                "linewidth": 3.0 if color.upper() == PALETTE_MIST.upper() else 2.2,
                "markersize": 8,
                "label": label,
            }
            if color.upper() == PALETTE_MIST.upper():
                kw["markerfacecolor"] = color
                kw["markeredgecolor"] = PALETTE_DEEP
                kw["markeredgewidth"] = 1.0
            ax.plot(x, y, **kw)
            for xi, yi in zip(x, y):
                if np.isfinite(yi):
                    ax.text(
                        xi,
                        yi,
                        f"{yi:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=FONT_ANNOT,
                        fontweight="bold",
                        color=annot_c,
                    )

        _style_subplot_axis(
            ax,
            title=f"{letters[ax_idx]} batch = {batch}",
            xlabel="input tokens",
            ylabel="tokens/s",
        )
        ax.set_xticks(INPUT_ORDER)
        ax.set_xticklabels([str(t) for t in INPUT_ORDER])
        ax.yaxis.grid(True, linestyle="--", color="grey", alpha=0.5)
        ax.set_axisbelow(True)

        if ax_idx == 0:
            ax.legend(
                loc="lower center",
                bbox_to_anchor=(0.5, 1.18),
                ncol=3,
                fontsize=FONT_LEGEND,
                prop={"weight": "bold"},
                frameon=True,
                fancybox=False,
                edgecolor="black",
            )

    _save_figure_both_formats(fig, out_path)
    plt.close(fig)


def figure_ttft_gpu_dual(df: pd.DataFrame, out_path: str) -> None:
    """Figure 2: grouped TTFT bars + GPU utilization lines (twin axis); solid fills, no hatch."""
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE_WIDE, dpi=DPI)
    fig.subplots_adjust(top=0.78, bottom=0.18, wspace=0.38)

    letters = ["(a1)", "(b1)", "(c1)"]
    bar_w = 0.22
    x_cat = np.arange(len(INPUT_ORDER), dtype=float)

    for ax_idx, batch in enumerate(BATCHES):
        ax_bar = axes[ax_idx]
        ax_gpu = ax_bar.twinx()
        sub = df[(df["batch"] == batch) & df["config"].isin([s[0] for s in SERIES])]

        for i, (cfg_key, label, color, marker, annot_c) in enumerate(SERIES):
            ttft_vals = []
            util_vals = []
            for it in INPUT_ORDER:
                row = sub[(sub["config"] == cfg_key) & (sub["input_tokens"] == int(it))]
                if row.empty:
                    ttft_vals.append(np.nan)
                    util_vals.append(np.nan)
                else:
                    ttft_vals.append(float(row["ttft_ms"].iloc[0]))
                    util_vals.append(float(row["gpu_util_pct"].iloc[0]))

            offsets = (i - 1) * bar_w
            xpos = x_cat + offsets
            bars = ax_bar.bar(
                xpos,
                ttft_vals,
                width=bar_w * 0.92,
                color=color,
                edgecolor="black",
                linewidth=0.35,
                alpha=0.88,
                label=label,
            )
            for rect, v in zip(bars, ttft_vals):
                if np.isfinite(v):
                    ax_bar.text(
                        rect.get_x() + rect.get_width() / 2,
                        rect.get_height(),
                        f"{v:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=FONT_ANNOT,
                        fontweight="bold",
                        color=annot_c,
                    )

            plot_kw: dict = {
                "color": color,
                "linestyle": "-",
                "marker": marker,
                "linewidth": 2.6 if color.upper() == PALETTE_MIST.upper() else 2.0,
                "markersize": 7,
            }
            if color.upper() == PALETTE_MIST.upper():
                plot_kw["markerfacecolor"] = color
                plot_kw["markeredgecolor"] = PALETTE_DEEP
                plot_kw["markeredgewidth"] = 0.9
            ax_gpu.plot(x_cat, util_vals, **plot_kw)

        _style_subplot_axis(
            ax_bar,
            title=f"{letters[ax_idx]} batch = {batch}",
            xlabel="Input Tokens",
            ylabel="TTFT (ms)",
            ylabel_color=PALETTE_DEEP,
        )
        ax_bar.set_xticks(x_cat)
        ax_bar.set_xticklabels([str(t) for t in INPUT_ORDER])
        ax_bar.tick_params(axis="y", labelcolor=PALETTE_DEEP)
        ax_bar.set_ylim(bottom=0)
        ax_bar.yaxis.grid(True, linestyle="--", color="grey", alpha=0.5)
        ax_bar.set_axisbelow(True)

        try:
            from cove_paths import env_str as _env_str

            gpu_lbl = _env_str("COVE_PLOT_GPU_YLABEL", "CVEE_PLOT_GPU_YLABEL") or "GPU utilization (%)"
        except ImportError:
            gpu_lbl = os.environ.get("COVE_PLOT_GPU_YLABEL") or os.environ.get("CVEE_PLOT_GPU_YLABEL") or "GPU utilization (%)"
        ax_gpu.set_ylabel(
            gpu_lbl,
            fontsize=FONT_LABEL,
            fontweight="bold",
            color=PALETTE_TEAL,
        )
        ax_gpu.tick_params(axis="y", labelsize=FONT_TICK, labelcolor=PALETTE_TEAL)
        ax_gpu.set_ylim(0, 100)

        if ax_idx == 0:
            ax_bar.legend(
                loc="lower center",
                bbox_to_anchor=(0.5, 1.18),
                ncol=3,
                fontsize=FONT_LEGEND,
                prop={"weight": "bold"},
                frameon=True,
                fancybox=False,
                edgecolor="black",
            )

    _save_figure_both_formats(fig, out_path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=os.path.join(_REPO, "eval-results", "llama-2-7b-hf"))
    ap.add_argument("--out-dir", default=None, help="Default: plotting/<model-slug>/ inferred from --results-dir")
    ap.add_argument("--model-ref", default=None, help="Optional model path to pick plotting output subdir")
    args = ap.parse_args()

    out_dir = args.out_dir or default_plotting_out_dir(args.results_dir, args.model_ref)
    os.makedirs(out_dir, exist_ok=True)
    raw = _prepare_df(args.results_dir)
    df = augment_interpolated_inputs(raw)

    p1 = os.path.join(out_dir, "fig_throughput_by_batch_16x9.png")
    p2 = os.path.join(out_dir, "fig_ttft_gpu_util_dual_16x9.png")
    figure_throughput_lines(df, p1)
    figure_ttft_gpu_dual(df, p2)
    print(
        "Wrote (PNG + PDF):\n"
        f"  {p1}\n  {os.path.splitext(p1)[0] + '.pdf'}\n"
        f"  {p2}\n  {os.path.splitext(p2)[0] + '.pdf'}"
    )


if __name__ == "__main__":
    main()
