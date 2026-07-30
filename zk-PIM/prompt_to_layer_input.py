import argparse
import os

import torch
import numpy as np

try:
    from . import fileio_utils
    from .model_load_utils import resolve_model_ref, load_tokenizer_and_model
except ImportError:
    import fileio_utils
    from model_load_utils import resolve_model_ref, load_tokenizer_and_model


def _load_low_freq_ids(path: str) -> set[int]:
    low_freq: set[int] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            # tolerate either "123" or "123\t..." formats
            tok = s.split()[0]
            try:
                low_freq.add(int(tok))
            except ValueError:
                # ignore malformed lines
                continue
    return low_freq


def _load_nearest_index_npz(path: str):
    """
    Load a compact nearest-neighbor index stored as NPZ.
    Expected keys:
      - neighbors: int32 [vocab_size, k]
      - scores: float32 [vocab_size, k]   (higher is better; e.g., cosine similarity)
    Uses mmap to avoid huge RAM spikes.
    """
    data = np.load(path, allow_pickle=False, mmap_mode="r")
    if "neighbors" not in data or "scores" not in data:
        raise ValueError(f"Bad npz schema: expected keys neighbors/scores in {path}")
    neighbors = data["neighbors"]
    scores = data["scores"]
    if neighbors.ndim != 2 or scores.ndim != 2 or neighbors.shape != scores.shape:
        raise ValueError(f"Bad npz shapes: neighbors {neighbors.shape} scores {scores.shape}")
    return neighbors, scores


def _lap_noise(score: float, epsilon: float, sensitivity: float) -> float:
    b = sensitivity / epsilon
    return float(score + np.random.laplace(loc=0.0, scale=b, size=1)[0])


def _gauss_noise(score: float, noise_factor: float = 2.60, mean: float = 0.0, std: float = 1.0) -> float:
    return float(score + noise_factor * np.random.normal(loc=mean, scale=std, size=1)[0])


def privatize_input_ids(
    input_ids: list[int],
    *,
    low_freq_ids: set[int],
    nearest_neighbors: np.ndarray,
    nearest_scores: np.ndarray,
    epsilon: float,
    sensitivity: float,
    noise_type: str,
    replace_prob_non_low_freq: float,
    gauss_noise_factor: float,
) -> list[int]:
    """
    DP sensitive-information-removal token replacement.
    - For low-frequency tokens: always replace with best noisy candidate from nearest set.
    - For other tokens: replace with probability `replace_prob_non_low_freq`.
    """
    out: list[int] = []
    vocab_size = int(nearest_neighbors.shape[0])
    for tid in input_ids:
        tid = int(tid)
        if tid < 0 or tid >= vocab_size:
            out.append(tid)
            continue
        do_replace = (tid in low_freq_ids) or (np.random.random() < replace_prob_non_low_freq)
        if not do_replace:
            out.append(tid)
            continue

        R = nearest_neighbors[tid]
        scores = nearest_scores[tid]
        if R.size == 0:
            out.append(tid)
            continue

        if noise_type == "gaussian":
            noisy = np.asarray([_gauss_noise(float(s), noise_factor=gauss_noise_factor) for s in scores], dtype=np.float64)
        else:
            noisy = np.asarray([_lap_noise(float(s), epsilon=epsilon, sensitivity=sensitivity) for s in scores], dtype=np.float64)

        out.append(int(R[int(np.argmax(noisy))]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_size", type=int, choices=[7, 13])
    ap.add_argument("--model_path", type=str, default=None, required=True,
                    help="Local HF-format model directory (e.g. /home/data/models/Llama-2-7b-hf)")
    ap.add_argument("--cache_dir", type=str, default="./model-storage")
    ap.add_argument("--prompt", type=str, required=True)
    ap.add_argument("--layer_number", type=int, default=0,
                    help="Which layer's input hidden states to export. 0 = embedding output (input to layer0).")
    ap.add_argument("--seq_len", type=int, default=None,
                    help="If set, pad/truncate tokenized prompt to this length")
    ap.add_argument("--output_file", type=str, default="layer_input.bin")
    ap.add_argument("--log_sf", type=int, default=16,
                    help="log2(scaling_factor); default 16 => scaling_factor=65536")

    # DP sensitive-information-removal token replacement.
    ap.add_argument("--dp_enable", action="store_true", help="Enable DP token replacement before embedding.")
    ap.add_argument("--dp_epsilon", type=float, default=100.0, help="Epsilon for Laplace noise (if enabled).")
    ap.add_argument("--dp_sensitivity", type=float, default=1.0, help="Sensitivity for Laplace noise (if enabled).")
    ap.add_argument("--dp_noise_type", type=str, choices=["laplace", "gaussian"], default="laplace")
    ap.add_argument("--dp_replace_prob", type=float, default=0.3,
                    help="For non-low-freq tokens, probability to apply replacement.")
    ap.add_argument("--dp_gauss_noise_factor", type=float, default=2.60,
                    help="Gaussian noise factor used by the DP helpers (if noise_type=gaussian).")
    ap.add_argument("--low_freq_words_txt", type=str,
                    default="PRISM/llama-2-7b-hf/low_freq_words.txt",
                    help="Path to low_freq_words.txt containing token ids (one per line).")
    ap.add_argument("--nearest_tokens_npz", type=str,
                    default="PRISM/llama-2-7b-hf/nearest_tokens_30.npz",
                    help="Path to nearest_tokens_30.npz (recommended; mmap-friendly).")
    ap.add_argument("--nearest_tokens_json", type=str, default=None,
                    help="(Not recommended) Path to nearest_tokens_30.json. Will be rejected if file is too large.")
    ap.add_argument("--allow_large_json", action="store_true",
                    help="Allow loading nearest_tokens_30.json even if it's very large (may OOM).")
    args = ap.parse_args()

    if not os.path.isdir(args.model_path):
        raise ValueError(f"--model_path is not a directory: {args.model_path}")

    # Load model/tokenizer locally
    model_ref = resolve_model_ref(args.model_size, args.model_path)
    tokenizer, model = load_tokenizer_and_model(
        model_ref, cache_dir=args.cache_dir, local_files_only=True
    )

    device = torch.device("cuda:0")
    model.to(device)
    model.eval()

    # Tokenize
    enc = tokenizer(
        args.prompt,
        return_tensors="pt",
        add_special_tokens=True,
    )
    input_ids = enc["input_ids"].to(device)  # [1, L]

    # Optional pad/truncate to seq_len
    if args.seq_len is not None:
        L = input_ids.shape[1]
        if L < args.seq_len:
            pad_id = tokenizer.pad_token_id
            if pad_id is None:
                # LLaMA tokenizer often has no pad_token by default; use eos as pad.
                pad_id = tokenizer.eos_token_id
            pad = torch.full((1, args.seq_len - L), pad_id, device=device, dtype=input_ids.dtype)
            input_ids = torch.cat([input_ids, pad], dim=1)
        elif L > args.seq_len:
            input_ids = input_ids[:, : args.seq_len]

    # Optional DP sensitive-information-removal token replacement.
    if args.dp_enable:
        if not args.low_freq_words_txt:
            raise ValueError("--dp_enable requires --low_freq_words_txt")
        low_freq_ids = _load_low_freq_ids(args.low_freq_words_txt)

        if args.nearest_tokens_npz and os.path.isfile(args.nearest_tokens_npz):
            nearest_neighbors, nearest_scores = _load_nearest_index_npz(args.nearest_tokens_npz)
        elif args.nearest_tokens_json:
            # Safety: huge monolithic JSON dict is not stream-friendly and will likely OOM.
            try:
                size_bytes = os.path.getsize(args.nearest_tokens_json)
            except OSError:
                size_bytes = None
            if (not args.allow_large_json) and (size_bytes is None or size_bytes > 256 * 1024 * 1024):
                raise ValueError(
                    "nearest_tokens_30.json is too large to load safely. "
                    "Please generate/use nearest_tokens_30.npz (recommended). "
                    "If you really want JSON, pass --allow_large_json (may OOM)."
                )
            import json
            with open(args.nearest_tokens_json, "r", encoding="utf-8") as f:
                nearest_tokens = json.load(f)
            # Convert dict -> dense arrays (still memory-heavy).
            vocab_size = max(int(k) for k in nearest_tokens.keys()) + 1
            k = len(next(iter(nearest_tokens.values())))
            nearest_neighbors = np.zeros((vocab_size, k), dtype=np.int32)
            nearest_scores = np.zeros((vocab_size, k), dtype=np.float32)
            for ks, items in nearest_tokens.items():
                tid = int(ks)
                nearest_neighbors[tid] = [int(it[0]) for it in items]
                nearest_scores[tid] = [float(it[1]) for it in items]
        else:
            raise ValueError(
                "--dp_enable requires either an existing --nearest_tokens_npz "
                "or a (small) --nearest_tokens_json."
            )

        ids_list = input_ids[0].tolist()
        private_ids = privatize_input_ids(
            ids_list,
            low_freq_ids=low_freq_ids,
            nearest_neighbors=nearest_neighbors,
            nearest_scores=nearest_scores,
            epsilon=args.dp_epsilon,
            sensitivity=args.dp_sensitivity,
            noise_type=args.dp_noise_type,
            replace_prob_non_low_freq=args.dp_replace_prob,
            gauss_noise_factor=args.dp_gauss_noise_factor,
        )
        input_ids = torch.tensor([private_ids], device=device, dtype=input_ids.dtype)

    # Export "layer input" tensor for zkLLM pipeline.
    # layer_number=0 is embedding output (input to layer0 rmsnorm).
    with torch.no_grad():
        if args.layer_number == 0:
            # model.model.embed_tokens: [bs, seq, dim]
            hidden = model.model.embed_tokens(input_ids)
        else:
            # WARNING: This computes a full forward pass to collect hidden states.
            # The input to layer N is hidden_states[N], where hidden_states[0] is embeddings.
            out = model(
                input_ids=input_ids,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            hs = out.hidden_states
            if hs is None or args.layer_number >= len(hs):
                raise RuntimeError(f"hidden_states unavailable or layer_number out of range: {args.layer_number}")
            hidden = hs[args.layer_number]

    # Convert to [seq, dim]
    hidden = hidden[0].contiguous()

    scaling_factor = 1 << args.log_sf
    fileio_utils.save_int(hidden, scaling_factor, args.output_file)

    print(f"Wrote {args.output_file}")
    print(f"shape={tuple(hidden.shape)} scaling_factor={scaling_factor} layer_number={args.layer_number} dp_enable={args.dp_enable}")


if __name__ == "__main__":
    main()