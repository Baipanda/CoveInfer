"""Build nearest_tokens K npz for any local HF model (GPU batched cosine; no faiss)."""
import argparse
import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import cove_paths  # noqa: F401  — registers zkllm → zk-PIM import alias
from zkllm.model_load_utils import load_tokenizer_and_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--cache_dir", default=os.path.join(REPO_ROOT, "model-storage"))
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--output_npz", required=True)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    _, model = load_tokenizer_and_model(args.model_path, cache_dir=args.cache_dir, local_files_only=True)
    # Keep only the embedding matrix; loading the full 7B weights on GPU OOMs.
    E = model.model.embed_tokens.weight.detach().float().cpu()
    del model
    if device.type == "cuda":
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    E = E.to(device)
    E = E / (E.norm(dim=1, keepdim=True) + 1e-12)
    V, D = E.shape
    print(f"vocab={V} dim={D} k={args.k}")

    k = int(args.k)
    neighbors = np.zeros((V, k), dtype=np.int32)
    scores = np.zeros((V, k), dtype=np.float32)
    batch = max(1, int(args.batch))
    if device.type != "cuda":
        batch = min(batch, 64)
    Et = E.T
    for start in range(0, V, batch):
        end = min(start + batch, V)
        sim = E[start:end] @ Et
        vals, idx = torch.topk(sim, k + 1, dim=1)
        neighbors[start:end] = idx[:, 1:].cpu().numpy().astype(np.int32)
        scores[start:end] = vals[:, 1:].cpu().numpy().astype(np.float32)
        if start == 0 or end == V or end % (batch * 40) == 0:
            print(f"  progress {end}/{V}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_npz)) or ".", exist_ok=True)
    np.savez_compressed(args.output_npz, neighbors=neighbors, scores=scores)
    print(f"Wrote {args.output_npz} neighbors={neighbors.shape}")


if __name__ == "__main__":
    main()
