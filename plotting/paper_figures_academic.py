#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Academic paper-style figures from eval-results JSONL.

Same metrics as paper_figures_16_9.py but with serif typography, grouped bars,
value labels, and full box spines. Uses the same 16:9 panel ratio and palette
as paper_figures_16_9.py. Outputs use distinct filenames so 16:9 figures are
never overwritten.

Usage:
  python plotting/paper_figures_academic.py --results-dir eval-results/llama-2-7b-hf
  python plotting/paper_figures_academic.py --results-dir eval-results/qwen2.5-7b-instruct
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from cove_paths import default_plotting_out_dir  # noqa: E402
from plotting.make_paper_artifacts import load_results  # noqa: E402
from plotting.paper_figures_16_9 import (  # noqa: E402
    INPUT_ORDER,
    BATCHES,
    PALETTE_DEEP,
    PALETTE_MIST,
    PALETTE_TEAL,
    augment_interpolated_inputs,
    _prepare_df,
    SUBPLOT_WIDTH_IN,
    SUBPLOT_HEIGHT_IN,
    SUBPLOT_BOX_ASPECT,
    DPI,
)

SERIES: list[tuple[str, str, str, str, str]] = [
    ("CSV+4090", "CSV + 4090", PALETTE_DEEP, "o", PALETTE_DEEP),
    ("CSV+5090", "CSV + 5090", PALETTE_TEAL, "s", PALETTE_DEEP),
    ("H20", "TDX + H20 (Baseline)", PALETTE_MIST, "^", PALETTE_DEEP),
]

# One row of three 16:9 panels; tighter horizontal packing.
FIGSIZE = (3 * SUBPLOT_WIDTH_IN + 0.45, SUBPLOT_HEIGHT_IN + 1.05)
WSPACE = 0.16
SUBPLOT_MARGINS = dict(left=0.06, right=0.995, top=0.86, bottom=0.24)

FONT_FAMILY = "serif"
FONT_LABEL = 11
FONT_TICK = 10
FONT_LEGEND = 9
FONT_ANNOT = 8
FONT_SUBCAP = 11


def _apply_academic_rc() -> None:
    plt.rcParams.update(
        {
            "font.family": FONT_FAMILY,
            "font.serif": ["Times New Roman", "DejaVu Serif", "Times", "serif"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "legend.framealpha": 1.0,
            "legend.edgecolor": "black",
            "legend.fancybox": False,
        }
    )


def _style_axis(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
    ax.yaxis.grid(True, linestyle="--", color="0.75", alpha=0.7, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=FONT_TICK, direction="out")
    ax.set_ylim(bottom=0)
    ax.set_box_aspect(SUBPLOT_BOX_ASPECT)


def _legend_upper_left(ax: plt.Axes) -> None:
    """Legend at the panel's top-left, sitting just above the plot area (no bar overlap)."""
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.01),
        borderaxespad=0.0,
        fontsize=FONT_LEGEND,
        frameon=True,
        fancybox=False,
        edgecolor="black",
        framealpha=1.0,
        borderpad=0.28,
        handlelength=1.4,
        labelspacing=0.25,
    )


def _subcaption(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.5,
        -0.22,
        label,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=FONT_SUBCAP,
        clip_on=False,
    )


def _save_both(fig: plt.Figure, png_path: str) -> None:
    fig.savefig(png_path, dpi=DPI, pad_inches=0.08)
    pdf_path = os.path.splitext(png_path)[0] + ".pdf"
    fig.savefig(pdf_path, format="pdf", pad_inches=0.08)


def _label_bars(ax: plt.Axes, bars, values: list[float], *, color: str = PALETTE_DEEP) -> None:
    for rect, v in zip(bars, values):
        if np.isfinite(v):
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                rect.get_height(),
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=FONT_ANNOT,
                color=color,
            )


def figure_throughput_bars(df: pd.DataFrame, out_path: str) -> None:
    """Grouped bar chart: throughput vs input tokens, one panel per batch."""
    _apply_academic_rc()
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE, dpi=DPI)
    fig.subplots_adjust(wspace=WSPACE, **SUBPLOT_MARGINS)

    letters = ["(a)", "(b)", "(c)"]
    n_series = len(SERIES)
    bar_w = 0.22
    x_cat = np.arange(len(INPUT_ORDER), dtype=float)

    for ax_idx, batch in enumerate(BATCHES):
        ax = axes[ax_idx]
        sub = df[(df["batch"] == batch) & df["config"].isin([s[0] for s in SERIES])]

        for i, (cfg_key, label, color, _marker, annot_c) in enumerate(SERIES):
            vals = []
            for it in INPUT_ORDER:
                row = sub[(sub["config"] == cfg_key) & (sub["input_tokens"] == int(it))]
                vals.append(float(row["tokens_per_s"].iloc[0]) if not row.empty else np.nan)

            xpos = x_cat + (i - (n_series - 1) / 2) * bar_w
            bars = ax.bar(
                xpos,
                vals,
                width=bar_w * 0.92,
                color=color,
                edgecolor="black",
                linewidth=0.6,
                label=label,
            )
            _label_bars(ax, bars, vals, color=annot_c)

        ax.set_xlabel("Input tokens", fontsize=FONT_LABEL)
        if ax_idx == 0:
            ax.set_ylabel("Throughput [tokens/s]", fontsize=FONT_LABEL)
        ax.set_xticks(x_cat)
        ax.set_xticklabels([str(t) for t in INPUT_ORDER])
        _style_axis(ax)
        _legend_upper_left(ax)
        _subcaption(ax, f"{letters[ax_idx]} batch = {batch}")

    _save_both(fig, out_path)
    plt.close(fig)


def figure_ttft_gpu_academic(df: pd.DataFrame, out_path: str) -> None:
    """
    TTFT grouped bars + GPU utilization trend lines (reference panel c style).
    """
    _apply_academic_rc()
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE, dpi=DPI)
    fig.subplots_adjust(wspace=WSPACE, **SUBPLOT_MARGINS)

    letters = ["(a)", "(b)", "(c)"]
    n_series = len(SERIES)
    bar_w = 0.22
    x_cat = np.arange(len(INPUT_ORDER), dtype=float)

    gpu_line_styles = ["--", "-.", ":", "--"]
    gpu_markers = ["s", "^", "D", "v"]

    for ax_idx, batch in enumerate(BATCHES):
        ax_bar = axes[ax_idx]
        ax_gpu = ax_bar.twinx()
        sub = df[(df["batch"] == batch) & df["config"].isin([s[0] for s in SERIES])]

        gpu_series: list[tuple[np.ndarray, np.ndarray, str, str, str]] = []

        for i, (cfg_key, label, color, _marker, annot_c) in enumerate(SERIES):
            ttft_vals: list[float] = []
            util_vals: list[float] = []
            for it in INPUT_ORDER:
                row = sub[(sub["config"] == cfg_key) & (sub["input_tokens"] == int(it))]
                if row.empty:
                    ttft_vals.append(np.nan)
                    util_vals.append(np.nan)
                else:
                    ttft_vals.append(float(row["ttft_ms"].iloc[0]))
                    util_vals.append(float(row["gpu_util_pct"].iloc[0]))

            xpos = x_cat + (i - (n_series - 1) / 2) * bar_w
            bars = ax_bar.bar(
                xpos,
                ttft_vals,
                width=bar_w * 0.92,
                color=color,
                edgecolor="black",
                linewidth=0.6,
                label=label,
            )
            _label_bars(ax_bar, bars, ttft_vals, color=annot_c)
            gpu_series.append(
                (xpos, np.array(util_vals, dtype=float), color, gpu_line_styles[i], gpu_markers[i])
            )

        for xpos, util, color, ls, mk in gpu_series:
            finite = np.isfinite(util)
            if not np.any(finite):
                continue
            plot_kw: dict = {
                "color": color,
                "linestyle": ls,
                "marker": mk,
                "markersize": 4.5,
                "linewidth": 1.2,
                "alpha": 0.95,
            }
            if color.upper() == PALETTE_MIST.upper():
                plot_kw["markerfacecolor"] = color
                plot_kw["markeredgecolor"] = PALETTE_DEEP
                plot_kw["markeredgewidth"] = 0.8
            ax_gpu.plot(xpos[finite], util[finite], **plot_kw)
            for xi, yi in zip(xpos[finite], util[finite]):
                ax_gpu.text(
                    xi,
                    yi,
                    f"{yi:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=FONT_ANNOT - 0.5,
                    color=PALETTE_DEEP,
                )

        ax_bar.set_xlabel("Input tokens", fontsize=FONT_LABEL)
        if ax_idx == 0:
            ax_bar.set_ylabel("TTFT [ms]", fontsize=FONT_LABEL)
        ax_bar.set_xticks(x_cat)
        ax_bar.set_xticklabels([str(t) for t in INPUT_ORDER])
        _style_axis(ax_bar)
        _legend_upper_left(ax_bar)

        try:
            from cove_paths import env_str as _env_str

            gpu_lbl = _env_str("COVE_PLOT_GPU_YLABEL", "CVEE_PLOT_GPU_YLABEL") or "GPU util. [%]"
        except ImportError:
            gpu_lbl = (
                os.environ.get("COVE_PLOT_GPU_YLABEL")
                or os.environ.get("CVEE_PLOT_GPU_YLABEL")
                or "GPU util. [%]"
            )
        ax_gpu.set_ylabel(gpu_lbl, fontsize=FONT_LABEL)
        ax_gpu.tick_params(axis="y", labelsize=FONT_TICK)
        ax_gpu.set_ylim(0, 100)
        ax_gpu.yaxis.grid(False)

        _subcaption(ax_bar, f"{letters[ax_idx]} batch = {batch}")

    _save_both(fig, out_path)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=os.path.join(_REPO, "eval-results", "llama-2-7b-hf"))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--model-ref", default=None)
    args = ap.parse_args()

    out_dir = args.out_dir or default_plotting_out_dir(args.results_dir, args.model_ref)
    os.makedirs(out_dir, exist_ok=True)

    raw = _prepare_df(args.results_dir)
    df = augment_interpolated_inputs(raw)

    p1 = os.path.join(out_dir, "fig_throughput_by_batch_academic.png")
    p2 = os.path.join(out_dir, "fig_ttft_gpu_util_dual_academic.png")
    figure_throughput_bars(df, p1)
    figure_ttft_gpu_academic(df, p2)
    print(
        "Wrote (PNG + PDF):\n"
        f"  {p1}\n  {os.path.splitext(p1)[0] + '.pdf'}\n"
        f"  {p2}\n  {os.path.splitext(p2)[0] + '.pdf'}"
    )


if __name__ == "__main__":
    main()
