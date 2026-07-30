"""
Preflight checks before invoking CUDA proof binaries (rmsnorm, ffn, etc.).
Catches missing commitments / wrong tensor file sizes early with clear errors.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from typing import Sequence


def _missing(paths: Sequence[str]) -> list[str]:
    return [p for p in paths if not os.path.isfile(p)]


def rmsnorm_weight_paths(workdir: str, layer_prefix: str, which: str) -> list[str]:
    w = workdir.rstrip("/")
    return [
        f"{w}/{which}_layernorm.weight-pp.bin",
        f"{w}/{layer_prefix}-{which}_layernorm.weight-int.bin",
        f"{w}/{layer_prefix}-{which}_layernorm.weight-commitment.bin",
    ]


def ffn_weight_paths(workdir: str, layer_prefix: str) -> list[str]:
    w = workdir.rstrip("/")
    names = [
        "mlp.up_proj.weight",
        "mlp.gate_proj.weight",
        "mlp.down_proj.weight",
    ]
    out: list[str] = []
    for n in names:
        out.extend(
            [
                f"{w}/{n}-pp.bin",
                f"{w}/{layer_prefix}-{n}-int.bin",
                f"{w}/{layer_prefix}-{n}-commitment.bin",
            ]
        )
    return out


def assert_input_int32_matrix(path: str, rows: int, cols: int, *, label: str) -> None:
    need = rows * cols * 4
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{label}: file not found: {path}")
    sz = os.path.getsize(path)
    if sz != need:
        raise ValueError(
            f"{label}: {path!r} is {sz} bytes; expected {need} (= {rows} x {cols} int32). "
            f"Check seq_len / embed_dim and that this file matches the pipeline stage."
        )


def run_native_or_die(
    argv: Sequence[str],
    *,
    cwd: str,
    name: str,
    extra_hint: str = "",
) -> None:
    """Run native binary; on failure print exit code and common fixes."""
    bin_path = argv[0]
    if not os.path.isfile(bin_path) and not shutil.which(bin_path):
        raise FileNotFoundError(f"Native binary not found: {bin_path} (cwd={cwd})")

    p = subprocess.run(list(argv), cwd=cwd)
    if p.returncode == 0:
        return

    sig = ""
    if p.returncode < 0:
        sig = f" (killed by OS signal {-p.returncode})"

    msg = (
        f"\n[{name}] native process exited with code {p.returncode}{sig}.\n"
        "Common causes:\n"
        "  1) Wrong GPU arch: run  cd zk-PIM && make clean && make   "
        "(Makefile auto-picks SM from nvidia-smi; override: make SM=90)\n"
        "  2) Missing or mismatched commitment files vs current model — "
        "re-run llama-ppgen.py and llama-commit.py for this model_path.\n"
        "  3) Shell variables empty: use literal args, e.g. "
        "python llama-rmsnorm.py 7 0 post_attention 2048 --input_file /path/in.bin ...\n"
    )
    if extra_hint:
        msg += f"  {extra_hint}\n"
    raise RuntimeError(msg.strip())
