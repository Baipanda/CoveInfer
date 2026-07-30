#!/usr/bin/env python3
"""Extract the 512/128/batch=1 table row from an eval-results JSONL file."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_row(path: Path) -> dict:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if (
                int(row.get("input_tokens", -1)) == 512
                and int(row.get("output_tokens", -1)) == 128
                and int(row.get("batch", -1)) == 1
            ):
                rows.append(row)
    if not rows:
        raise SystemExit(f"No 512/128/batch=1 row in {path}")
    return rows[-1]


def fmt(row: dict) -> str:
    h2d = float(row["transfer_ms"])
    ttft = float(row["ttft_ms"])
    total = float(row["total_ms"])
    decode = total - ttft
    gpu_sub = ttft + decode
    tps = float(row["tokens_per_s"])
    return (
        f"config={row.get('config')}\n"
        f"model_ref={row.get('model_ref')}\n"
        f"H2D={h2d:.1f}\n"
        f"TTFT={ttft:.1f}\n"
        f"Decode={decode:.1f}\n"
        f"GPU Sub.={gpu_sub:.1f}\n"
        f"E2E Total={gpu_sub:.1f}\n"
        f"Throughput={tps:.2f}\n"
        f"LaTeX row (H20 baseline): "
        f"--- & --- & {h2d:.1f} & {ttft:.1f} & {decode:.1f} & {gpu_sub:.1f} & {gpu_sub:.1f} & {tps:.2f}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", type=Path, help="eval-results JSONL file")
    args = ap.parse_args()
    print(fmt(load_row(args.jsonl)))


if __name__ == "__main__":
    main()
