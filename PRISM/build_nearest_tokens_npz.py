import argparse
import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import cove_paths  # noqa: F401  — registers zkllm → zk-PIM import alias
from zkllm.model_load_utils import resolve_model_ref, load_tokenizer_and_model


def main():
    ap = argparse.ArgumentParser(description="Build nearest_tokens_30.npz from model embeddings (HF format).")
    ap.add_argument("model_size", type=int, choices=[7, 13])
    ap.add_argument("--model_path", type=str, required=True, help="Local HF-format model directory.")
    ap.add_argument("--cache_dir", type=str, default="./model-storage")
    ap.add_argument("--k", type=int, default=30, help="Number of neighbors per token.")
    ap.add_argument("--metric", type=str, choices=["cosine"], default="cosine")
    ap.add_argument("--output_npz", type=str, default="PRISM/llama-2-7b-hf/nearest_tokens_30.npz")
    ap.add_argument("--method", type=str, choices=["hnsw", "flat"], default="hnsw",
                    help="Nearest neighbor backend. flat is exact but very slow; hnsw is fast approximate.")
    ap.add_argument("--hnsw_m", type=int, default=32, help="HNSW graph degree (bigger = better/slow).")
    ap.add_argument("--hnsw_ef_construction", type=int, default=200, help="HNSW construction ef.")
    ap.add_argument("--hnsw_ef_search", type=int, default=128, help="HNSW search ef (bigger = better/slow).")
    args = ap.parse_args()

    model_ref = resolve_model_ref(args.model_size, args.model_path)
    _, model = load_tokenizer_and_model(model_ref, cache_dir=args.cache_dir, local_files_only=True)

    # Get token embedding matrix [vocab, dim]
    with torch.no_grad():
        E = model.model.embed_tokens.weight.detach().float().cpu().numpy()

    # Normalize for cosine similarity
    if args.metric == "cosine":
        norms = np.linalg.norm(E, axis=1, keepdims=True) + 1e-12
        E = E / norms

    # Use FAISS if available (recommended; fast + memory-friendly).
    try:
        import faiss  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "faiss is required to build nearest_tokens_30.npz efficiently.\n"
            "Install one of:\n"
            "  - pip install faiss-cpu\n"
            "  - pip install faiss-gpu\n"
            f"Original import error: {e}"
        )

    dim = E.shape[1]
    if args.method == "flat":
        index = faiss.IndexFlatIP(dim)  # exact, but O(V^2) when querying all vectors
    else:
        # Fast approximate search for large V
        index = faiss.IndexHNSWFlat(dim, args.hnsw_m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = args.hnsw_ef_construction
        index.hnsw.efSearch = args.hnsw_ef_search

    index.add(E.astype(np.float32))

    # Query all vectors against the index (k+1 to skip self)
    scores, neighbors = index.search(E.astype(np.float32), args.k + 1)
    neighbors = neighbors[:, 1 : args.k + 1].astype(np.int32)
    scores = scores[:, 1 : args.k + 1].astype(np.float32)

    os.makedirs(os.path.dirname(args.output_npz) or ".", exist_ok=True)
    np.savez_compressed(args.output_npz, neighbors=neighbors, scores=scores)
    print(f"Wrote {args.output_npz} neighbors={neighbors.shape} scores={scores.shape}")


if __name__ == "__main__":
    main()

