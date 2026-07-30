# Plotting

Turn `eval-results/<model-slug>/*.jsonl` into paper figures and LaTeX tables.
Outputs go to `plotting/<model-slug>/` (one folder per model).

## Llama-2-7B

```bash
python plotting/make_paper_artifacts.py --results-dir eval-results/llama-2-7b-hf
python plotting/paper_figures_16_9.py --results-dir eval-results/llama-2-7b-hf
python plotting/paper_figures_academic.py --results-dir eval-results/llama-2-7b-hf
python plotting/paper_figures_multi_style.py --results-dir eval-results/llama-2-7b-hf
```

## Qwen2.5-7B-Instruct

```bash
python plotting/make_paper_artifacts.py --results-dir eval-results/qwen2.5-7b-instruct
python plotting/paper_figures_16_9.py --results-dir eval-results/qwen2.5-7b-instruct
python plotting/paper_figures_academic.py --results-dir eval-results/qwen2.5-7b-instruct
python plotting/paper_figures_multi_style.py --results-dir eval-results/qwen2.5-7b-instruct
```

## Artifacts (per model subdir)

- `fig_ttft_vs_input.png`
- `fig_throughput_vs_batch.png`
- `fig_total_ms_vs_input.png`
- `fig_throughput_by_batch_16x9.png` / `.pdf`
- `fig_ttft_gpu_util_dual_16x9.png` / `.pdf`
- `fig_throughput_by_batch_academic.png` / `.pdf` (serif / pastel paper style)
- `fig_ttft_gpu_util_dual_academic.png` / `.pdf`

## Multi-style variants (`plotting/style-variants/<model>/`)

```bash
python plotting/paper_figures_multi_style.py --results-dir eval-results/llama-2-7b-hf
```

Subfolders (each with throughput + TTFT/GPU dual figures, PNG + PDF):

| Folder | Method |
|--------|--------|
| `ieee/` | Matplotlib IEEE-like serif |
| `nature/` | Matplotlib minimal (Nature-ish) |
| `seaborn-paper/` | Seaborn white paper context |
| `seaborn-ticks/` | Seaborn ticks style |
| `line-clean/` | Matplotlib line charts (16:9) |
| `colorbrewer/` | ColorBrewer palette comparison |
- `table_summary.tex`
