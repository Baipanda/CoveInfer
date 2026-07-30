import argparse
import collections
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import cove_paths  # noqa: F401  — registers zkllm → zk-PIM import alias
from zkllm.model_load_utils import resolve_model_ref, load_tokenizer_and_model


def iter_lines(path: str):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if s:
                yield s


def main():
    ap = argparse.ArgumentParser(description="Build low_freq_words.txt by counting token ids on your own corpus.")
    ap.add_argument("model_size", type=int, choices=[7, 13])
    ap.add_argument("--model_path", type=str, required=True, help="Local HF-format model directory.")
    ap.add_argument("--cache_dir", type=str, default="./model-storage")
    ap.add_argument("--corpus_txt", type=str, required=True, help="A plain text file (one sample per line).")
    ap.add_argument("--num_low_freq", type=int, default=6400)
    ap.add_argument("--output_txt", type=str, default="PRISM/llama-2-7b-hf/low_freq_words.txt")
    ap.add_argument("--max_lines", type=int, default=200000, help="Limit lines for speed; 0 means no limit.")
    args = ap.parse_args()

    model_ref = resolve_model_ref(args.model_size, args.model_path)
    tokenizer, _ = load_tokenizer_and_model(model_ref, cache_dir=args.cache_dir, local_files_only=True)

    counter = collections.Counter()
    n = 0
    for line in iter_lines(args.corpus_txt):
        ids = tokenizer(line, add_special_tokens=False, return_tensors=None)["input_ids"]
        counter.update(ids)
        n += 1
        if args.max_lines and n >= args.max_lines:
            break

    # Exclude special tokens if known
    special = set()
    for attr in ["pad_token_id", "eos_token_id", "bos_token_id", "unk_token_id"]:
        tid = getattr(tokenizer, attr, None)
        if tid is not None:
            special.add(int(tid))

    # Pick lowest-frequency token ids
    items = [(tid, c) for tid, c in counter.items() if int(tid) not in special]
    items.sort(key=lambda x: (x[1], x[0]))
    low = [int(tid) for tid, _ in items[: args.num_low_freq]]

    with open(args.output_txt, "w", encoding="utf-8") as f:
        for tid in low:
            f.write(f"{tid}\n")

    print(f"Wrote {args.output_txt} ({len(low)} ids) from {n} lines")


if __name__ == "__main__":
    main()

