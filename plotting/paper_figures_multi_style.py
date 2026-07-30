#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate the same benchmark figures in multiple paper-ready styles.

Outputs under plotting/style-variants/<model-slug>/<style-name>/ so existing
figures are never overwritten.

Styles (best-effort; skips unavailable backends):
  - ieee          Matplotlib, IEEE-like serif + boxed panels
  - nature        Matplotlib, minimal spines (Nature-ish)
  - seaborn-paper Seaborn paper context + grouped bars
  - seaborn-ticks Seaborn ticks style, muted grid
  - line-clean    Matplotlib line charts (16:9), shared legend row
  - colorbrewer   Matplotlib, ColorBrewer-inspired palette (comparison)

Usage:
  python plotting/paper_figures_multi_style.py --results-dir eval-results/llama-2-7b-hf
  python plotting/paper_figures_multi_style.py --results-dir eval-results/qwen2.5-7b-instruct
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from cove_paths import default_plotting_out_dir  # noqa: E402
from plotting.paper_figures_16_9 import (  # noqa: E402
    BATCHES,
    DPI,
    INPUT_ORDER,
    PALETTE_DEEP,
    PALETTE_MIST,
    PALETTE_TEAL,
    SUBPLOT_BOX_ASPECT,
    SUBPLOT_HEIGHT_IN,
    SUBPLOT_WIDTH_IN,
    augment_interpolated_inputs,
    _prepare_df,
)

try:
    import seaborn as sns

    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

STYLE_VARIANTS_ROOT = os.path.join(_REPO, "plotting", "style-variants")

FIGSIZE = (3 * SUBPLOT_WIDTH_IN + 0.45, SUBPLOT_HEIGHT_IN + 1.05)
WSPACE = 0.14
MARGINS = dict(left=0.06, right=0.995, top=0.86, bottom=0.24)

# Cove palette (default for most styles)
COVE_SERIES: list[tuple[str, str, str, str, str]] = [
    ("CSV+4090", "CSV + 4090", PALETTE_DEEP, "o", PALETTE_DEEP),
    ("CSV+5090", "CSV + 5090", PALETTE_TEAL, "s", PALETTE_DEEP),
    ("H20", "TDX + H20 (Baseline)", PALETTE_MIST, "^", PALETTE_DEEP),
]

# Alternate palette for side-by-side comparison
BREWER_SERIES: list[tuple[str, str, str, str, str]] = [
    ("CSV+4090", "CSV + 4090", "#3182bd", "o", "#1f3a5f"),
    ("CSV+5090", "CSV + 5090", "#31a354", "s", "#1f3a5f"),
    ("H20", "TDX + H20 (Baseline)", "#756bb1", "^", "#1f3a5f"),
]


@dataclass(frozen=True)
class StyleSpec:
    name: str
    description: str
    series: list[tuple[str, str, str, str, str]]
    apply_rc: Callable[[], None]
    bar_edge: str
    bar_lw: float
    spine_mode: str  # "full" | "minimal"
    grid_alpha: float
    font_label: int = 11
    font_tick: int = 10
    font_legend: int = 9
    font_annot: int = 8
    font_subcap: int = 11
    annot_bold: bool = False
    axis_bold: bool = False
    legend_anchor_y: float = 1.01
    legend_placement: str = "subplot_upper"  # subplot_upper | figure_lower_left
    legend_anchor_x: float = 0.06
    legend_anchor_y: float = 0.02
    throughput_kind: str = "bar"  # "bar" | "line"


def _reset_matplotlib() -> None:
    plt.rcdefaults()
    plt.rcParams["axes.unicode_minus"] = False


def _apply_ieee() -> None:
    _reset_matplotlib()
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Times"],
            "mathtext.fontset": "stix",
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "legend.framealpha": 1.0,
            "legend.edgecolor": "black",
            "legend.fancybox": False,
        }
    )


def _apply_nature() -> None:
    _reset_matplotlib()
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.linewidth": 0.6,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.frameon": False,
        }
    )


def _apply_seaborn_paper() -> None:
    _reset_matplotlib()
    if HAS_SEABORN:
        sns.set_theme(style="white", context="paper", font_scale=1.28)
        sns.set_palette([PALETTE_DEEP, PALETTE_TEAL, PALETTE_MIST])


def _apply_seaborn_ticks() -> None:
    _reset_matplotlib()
    if HAS_SEABORN:
        sns.set_theme(style="ticks", context="paper", font_scale=1.0)


def _apply_line_clean() -> None:
    _apply_ieee()


def _apply_colorbrewer() -> None:
    _apply_ieee()


STYLES: list[StyleSpec] = [
    StyleSpec(
        "ieee",
        "IEEE-like serif, full box, dashed grid",
        COVE_SERIES,
        _apply_ieee,
        bar_edge="black",
        bar_lw=0.6,
        spine_mode="full",
        grid_alpha=0.65,
    ),
    StyleSpec(
        "nature",
        "Minimal spines, sans-serif, light grid",
        COVE_SERIES,
        _apply_nature,
        bar_edge="0.35",
        bar_lw=0.4,
        spine_mode="minimal",
        grid_alpha=0.35,
    ),
    StyleSpec(
        "seaborn-paper",
        "Seaborn white paper context; single legend at figure lower-left",
        COVE_SERIES,
        _apply_seaborn_paper,
        bar_edge="0.25",
        bar_lw=0.5,
        spine_mode="minimal",
        grid_alpha=0.4,
        font_label=16,
        font_tick=16,
        font_legend=16,
        font_annot=13,
        font_subcap=16,
        annot_bold=True,
        axis_bold=True,
        legend_placement="figure_lower_left",
        legend_anchor_x=0.06,
        legend_anchor_y=0.04,
    ),
    StyleSpec(
        "seaborn-ticks",
        "Seaborn ticks style",
        COVE_SERIES,
        _apply_seaborn_ticks,
        bar_edge="0.2",
        bar_lw=0.45,
        spine_mode="minimal",
        grid_alpha=0.45,
    ),
    StyleSpec(
        "line-clean",
        "16:9 line charts, serif",
        COVE_SERIES,
        _apply_line_clean,
        bar_edge="black",
        bar_lw=0.6,
        spine_mode="full",
        grid_alpha=0.55,
        throughput_kind="line",
    ),
    StyleSpec(
        "colorbrewer",
        "ColorBrewer palette comparison",
        BREWER_SERIES,
        _apply_colorbrewer,
        bar_edge="0.15",
        bar_lw=0.5,
        spine_mode="full",
        grid_alpha=0.6,
    ),
]


def _style_spines(ax: plt.Axes, mode: str) -> None:
    if mode == "minimal":
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(0.6)
        ax.spines["bottom"].set_linewidth(0.6)
    else:
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_linewidth(0.8)


def _axis_label_kw(spec: StyleSpec) -> dict:
    kw: dict = {"fontsize": spec.font_label}
    if spec.axis_bold:
        kw["fontweight"] = "bold"
    return kw


def _apply_tick_style(ax: plt.Axes, spec: StyleSpec) -> None:
    ax.tick_params(axis="both", labelsize=spec.font_tick, direction="out")
    if not spec.axis_bold:
        return
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontweight("bold")
        lab.set_fontsize(spec.font_tick)


def _style_axis(ax: plt.Axes, spec: StyleSpec) -> None:
    _style_spines(ax, spec.spine_mode)
    ax.yaxis.grid(True, linestyle="--", color="0.78", alpha=spec.grid_alpha, linewidth=0.55)
    ax.set_axisbelow(True)
    _apply_tick_style(ax, spec)
    ax.set_ylim(bottom=0)
    ax.set_box_aspect(SUBPLOT_BOX_ASPECT)


def _subplot_margins(spec: StyleSpec) -> dict:
    m = dict(MARGINS)
    if spec.legend_placement == "figure_lower_left":
        m["bottom"] = 0.32
        m["top"] = 0.95
    return m


def _legend_props(spec: StyleSpec) -> dict:
    frame = spec.spine_mode != "minimal"
    hl = 1.35 if spec.legend_placement == "figure_lower_left" else 1.05
    kw: dict = {
        "frameon": frame,
        "fancybox": False,
        "edgecolor": "black" if frame else "none",
        "framealpha": 1.0 if frame else 0.0,
        "borderpad": 0.28 if spec.legend_placement == "figure_lower_left" else 0.22,
        "handlelength": hl,
        "handleheight": 0.85 if spec.legend_placement == "figure_lower_left" else 0.7,
        "handletextpad": 0.55,
        "labelspacing": 0.28,
    }
    if spec.axis_bold:
        kw["prop"] = {"size": spec.font_legend, "weight": "bold"}
    else:
        kw["fontsize"] = spec.font_legend
    return kw


def _place_legend(fig: plt.Figure, ax: plt.Axes, spec: StyleSpec) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    if spec.legend_placement == "figure_lower_left":
        fig.legend(
            handles,
            labels,
            loc="lower left",
            bbox_to_anchor=(spec.legend_anchor_x, spec.legend_anchor_y),
            ncol=len(labels),
            **_legend_props(spec),
        )
    else:
        ax.legend(
            loc="lower left",
            bbox_to_anchor=(0.0, spec.legend_anchor_y),
            borderaxespad=0.0,
            **_legend_props(spec),
        )


def _series_label(label: str, *, ax_idx: int, spec: StyleSpec) -> str | None:
    if spec.legend_placement == "figure_lower_left" and ax_idx != 0:
        return None
    return label


def _subcaption(ax: plt.Axes, label: str, spec: StyleSpec) -> None:
    ax.text(
        0.5,
        -0.22,
        label,
        transform=ax.transAxes,
        ha="center",
        va="top",
        clip_on=False,
        **_axis_label_kw(spec),
    )


def _save_both(fig: plt.Figure, png_path: str) -> None:
    os.makedirs(os.path.dirname(png_path), exist_ok=True)
    fig.savefig(png_path, dpi=DPI, pad_inches=0.08)
    fig.savefig(os.path.splitext(png_path)[0] + ".pdf", format="pdf", pad_inches=0.08)


def _lookup(df: pd.DataFrame, *, batch: int, cfg: str, col: str, it: int) -> float:
    row = df[(df["batch"] == batch) & (df["config"] == cfg) & (df["input_tokens"] == int(it))]
    if row.empty:
        return float("nan")
    return float(row[col].iloc[0])


def _annot_text_kw(spec: StyleSpec) -> dict:
    kw: dict = {"fontsize": spec.font_annot}
    if spec.annot_bold:
        kw["fontweight"] = "bold"
    return kw


def _label_bars(ax: plt.Axes, bars, values: list[float], *, color: str, spec: StyleSpec) -> None:
    for rect, v in zip(bars, values):
        if np.isfinite(v):
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                rect.get_height(),
                f"{v:.2f}",
                ha="center",
                va="bottom",
                color=color,
                **_annot_text_kw(spec),
            )


def figure_throughput(df: pd.DataFrame, out_path: str, spec: StyleSpec) -> None:
    spec.apply_rc()
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE, dpi=DPI)
    fig.subplots_adjust(wspace=WSPACE, **_subplot_margins(spec))

    letters = ["(a)", "(b)", "(c)"]
    n_series = len(spec.series)
    bar_w = 0.22
    x_cat = np.arange(len(INPUT_ORDER), dtype=float)

    for ax_idx, batch in enumerate(BATCHES):
        ax = axes[ax_idx]
        sub = df[(df["batch"] == batch) & df["config"].isin([s[0] for s in spec.series])]

        if spec.throughput_kind == "line":
            for cfg_key, label, color, marker, annot_c in spec.series:
                d = sub[sub["config"] == cfg_key].copy()
                d = d[d["input_tokens"].isin(INPUT_ORDER)].sort_values("input_tokens")
                x = d["input_tokens"].to_numpy(dtype=float)
                y = d["tokens_per_s"].to_numpy(dtype=float)
                if len(x) == 0:
                    continue
                kw: dict = {
                    "color": color,
                    "marker": marker,
                    "linewidth": 2.4,
                    "markersize": 7,
                    "label": _series_label(label, ax_idx=ax_idx, spec=spec),
                }
                if color.upper() == PALETTE_MIST.upper():
                    kw["markerfacecolor"] = color
                    kw["markeredgecolor"] = PALETTE_DEEP
                    kw["markeredgewidth"] = 0.9
                ax.plot(x, y, **kw)
                for xi, yi in zip(x, y):
                    if np.isfinite(yi):
                        ax.text(
                            xi,
                            yi,
                            f"{yi:.2f}",
                            ha="center",
                            va="bottom",
                            color=annot_c,
                            **_annot_text_kw(spec),
                        )
        else:
            for i, (cfg_key, label, color, _marker, annot_c) in enumerate(spec.series):
                vals = [_lookup(sub, batch=batch, cfg=cfg_key, col="tokens_per_s", it=it) for it in INPUT_ORDER]
                xpos = x_cat + (i - (n_series - 1) / 2) * bar_w
                bars = ax.bar(
                    xpos,
                    vals,
                    width=bar_w * 0.92,
                    color=color,
                    edgecolor=spec.bar_edge,
                    linewidth=spec.bar_lw,
                    label=_series_label(label, ax_idx=ax_idx, spec=spec),
                )
                _label_bars(ax, bars, vals, color=annot_c, spec=spec)

        ax.set_xlabel("Input tokens", **_axis_label_kw(spec))
        if ax_idx == 0:
            ax.set_ylabel("Throughput [tokens/s]", **_axis_label_kw(spec))
        ax.set_xticks(x_cat if spec.throughput_kind != "line" else INPUT_ORDER)
        ax.set_xticklabels([str(t) for t in INPUT_ORDER])
        _style_axis(ax, spec)
        _subcaption(ax, f"{letters[ax_idx]} batch = {batch}", spec)

    _place_legend(fig, axes[0], spec)
    _save_both(fig, out_path)
    plt.close(fig)


def figure_ttft_gpu(df: pd.DataFrame, out_path: str, spec: StyleSpec) -> None:
    spec.apply_rc()
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE, dpi=DPI)
    fig.subplots_adjust(wspace=WSPACE, **_subplot_margins(spec))

    letters = ["(a)", "(b)", "(c)"]
    n_series = len(spec.series)
    bar_w = 0.22
    x_cat = np.arange(len(INPUT_ORDER), dtype=float)
    gpu_ls = ["--", "-.", ":", "--"]
    gpu_mk = ["s", "^", "D", "v"]

    for ax_idx, batch in enumerate(BATCHES):
        ax_bar = axes[ax_idx]
        ax_gpu = ax_bar.twinx()
        sub = df[(df["batch"] == batch) & df["config"].isin([s[0] for s in spec.series])]

        for i, (cfg_key, label, color, _marker, annot_c) in enumerate(spec.series):
            ttft = [_lookup(sub, batch=batch, cfg=cfg_key, col="ttft_ms", it=it) for it in INPUT_ORDER]
            util = [_lookup(sub, batch=batch, cfg=cfg_key, col="gpu_util_pct", it=it) for it in INPUT_ORDER]
            xpos = x_cat + (i - (n_series - 1) / 2) * bar_w
            bars = ax_bar.bar(
                xpos,
                ttft,
                width=bar_w * 0.92,
                color=color,
                edgecolor=spec.bar_edge,
                linewidth=spec.bar_lw,
                label=_series_label(label, ax_idx=ax_idx, spec=spec),
            )
            _label_bars(ax_bar, bars, ttft, color=annot_c, spec=spec)

            finite = np.isfinite(util)
            if np.any(finite):
                pkw: dict = {
                    "color": color,
                    "linestyle": gpu_ls[i],
                    "marker": gpu_mk[i],
                    "markersize": 4.5,
                    "linewidth": 1.2,
                }
                mist = spec.series[i][2].upper() == PALETTE_MIST.upper()
                if mist or spec.series[i][2].upper() == "#756BB1":
                    pkw["markerfacecolor"] = color
                    pkw["markeredgecolor"] = annot_c
                    pkw["markeredgewidth"] = 0.8
                ax_gpu.plot(xpos[finite], np.array(util)[finite], **pkw)
                for xi, yi in zip(xpos[finite], np.array(util)[finite]):
                    ax_gpu.text(
                        xi,
                        yi,
                        f"{yi:.1f}",
                        ha="center",
                        va="bottom",
                        color=annot_c,
                        **_annot_text_kw(spec),
                    )

        ax_bar.set_xlabel("Input tokens", **_axis_label_kw(spec))
        if ax_idx == 0:
            ax_bar.set_ylabel("TTFT [ms]", **_axis_label_kw(spec))
        ax_bar.set_xticks(x_cat)
        ax_bar.set_xticklabels([str(t) for t in INPUT_ORDER])
        _style_axis(ax_bar, spec)

        gpu_lbl = os.environ.get("COVE_PLOT_GPU_YLABEL") or "GPU util. [%]"
        ax_gpu.set_ylabel(gpu_lbl, **_axis_label_kw(spec))
        _apply_tick_style(ax_gpu, spec)
        ax_gpu.set_ylim(0, 100)
        if spec.spine_mode == "minimal":
            ax_gpu.spines["top"].set_visible(False)
        ax_gpu.yaxis.grid(False)

        _subcaption(ax_bar, f"{letters[ax_idx]} batch = {batch}", spec)

    _place_legend(fig, axes[0], spec)

    _save_both(fig, out_path)
    plt.close(fig)


def _style_out_dir(results_dir: str, model_ref: str | None, style_name: str) -> str:
    base = default_plotting_out_dir(results_dir, model_ref)
    slug = os.path.basename(base)
    return os.path.join(STYLE_VARIANTS_ROOT, slug, style_name)


def generate_all_styles(df: pd.DataFrame, results_dir: str, model_ref: str | None) -> list[str]:
    written: list[str] = []
    for spec in STYLES:
        if spec.name.startswith("seaborn") and not HAS_SEABORN:
            print(f"[skip] {spec.name}: seaborn not installed")
            continue
        out_dir = _style_out_dir(results_dir, model_ref, spec.name)
        os.makedirs(out_dir, exist_ok=True)
        readme = os.path.join(out_dir, "STYLE.txt")
        with open(readme, "w", encoding="utf-8") as f:
            f.write(f"{spec.name}\n{spec.description}\n")

        p1 = os.path.join(out_dir, "fig_throughput_by_batch.png")
        p2 = os.path.join(out_dir, "fig_ttft_gpu_util_dual.png")
        figure_throughput(df, p1, spec)
        figure_ttft_gpu(df, p2, spec)
        written.extend([p1, p2, readme])
        print(f"[ok] {spec.name} -> {out_dir}")
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description="Multi-style paper figure generator")
    ap.add_argument("--results-dir", default=os.path.join(_REPO, "eval-results", "llama-2-7b-hf"))
    ap.add_argument("--model-ref", default=None)
    ap.add_argument(
        "--styles",
        default="all",
        help="Comma-separated style names or 'all'",
    )
    args = ap.parse_args()

    raw = _prepare_df(args.results_dir)
    df = augment_interpolated_inputs(raw)

    if args.styles.strip().lower() != "all":
        wanted = {s.strip() for s in args.styles.split(",") if s.strip()}
        global STYLES
        STYLES = [s for s in STYLES if s.name in wanted]

    written = generate_all_styles(df, args.results_dir, args.model_ref)
    root = _style_out_dir(args.results_dir, args.model_ref, "")
    print(f"\nWrote {len(written)} artifacts under {os.path.dirname(root)}/")


if __name__ == "__main__":
    main()
