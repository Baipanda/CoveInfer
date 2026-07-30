import os, sys
import argparse
import subprocess
import torch
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

from preflight_native import (
    assert_input_int32_matrix,
    rmsnorm_weight_paths,
    run_native_or_die,
)

parser = argparse.ArgumentParser(description='LLaMa-2 Self-Attention')
parser.add_argument('model_size', type=int, choices = [7, 13], help='The size of the model to use. Default is 13')
parser.add_argument('layer', type=int, help='The layer to use for rmsnorm')
parser.add_argument('which', type=str, choices=['input', 'post_attention'], help='To use the input norm or the post-attention norm')
parser.add_argument('seq_len', type=int, help='The sequence length to use for rmsnorm')
parser.add_argument('--input_file', required = True, type=str, help='The input file to use for rmsnorm')
parser.add_argument('--output_file', default = 'llama-rmsnorm-output.bin', type=str, help='The output file to use for rmsnorm')
parser.add_argument('--model_path', type=str, default=None, help='Local path to a LLaMA-2 HF-format folder (overrides HF model id).')
parser.add_argument('--cache_dir', type=str, default='./model-storage', help='Transformers cache dir (used when loading from cache).')

from transformers import AutoModelForCausalLM
import fileio_utils
from model_load_utils import resolve_model_ref, load_tokenizer_and_model


if __name__ == '__main__':
    compilation_error = subprocess.call(["make", "rmsnorm"], cwd=SCRIPT_DIR)
    if compilation_error:
        print("Error compiling rmsnorm")
        exit(1)
    args = parser.parse_args()
    if not str(args.input_file).strip() or not str(args.output_file).strip():
        print(
            "ERROR: --input_file / --output_file look empty. "
            "Did you forget to set shell variables? Use literals, e.g.\n"
            "  python llama-rmsnorm.py 7 0 post_attention 2048 "
            "--input_file /tmp/post_attn.bin --output_file /tmp/ffn_in.bin --model_path ...",
            file=sys.stderr,
        )
        sys.exit(2)
    model_ref = resolve_model_ref(args.model_size, args.model_path)
    _, model = load_tokenizer_and_model(model_ref, cache_dir=args.cache_dir, local_files_only=True)
    layer = getattr(model.model.layers[0], f'{args.which}_layernorm')
    # print(layer.eps)
    # print(layer.variance_epsilon)
    # print(layer.weight)
    # for param in layer.parameters():
    #     print(param.shape)
    (embed_dim, ) = layer.weight.shape
    workdir = f'./zkllm-workdir/Llama-2-{args.model_size}b'
    layer_prefix = f'layer-{args.layer}'

    miss = [p for p in rmsnorm_weight_paths(workdir, layer_prefix, args.which) if not os.path.isfile(p)]
    if miss:
        print(
            "ERROR: native rmsnorm needs these files (missing):\n  "
            + "\n  ".join(miss)
            + "\nRun llama-ppgen.py then llama-commit.py for this model, or fix --model_path / workdir.",
            file=sys.stderr,
        )
        sys.exit(2)

    if not os.path.isfile(args.input_file):
        temp_X = torch.randn(args.seq_len, embed_dim, device = 0)
        fileio_utils.save_int(temp_X, 1 << 16, args.input_file)
    assert_input_int32_matrix(args.input_file, args.seq_len, embed_dim, label="rmsnorm input")
    X = torch.tensor(np.fromfile(args.input_file, dtype = np.int32).reshape(args.seq_len, embed_dim), device = 0, dtype = float) / (1 << 16)
    rms_inv = 1 / torch.sqrt(torch.mean(X ** 2, dim = 1) + layer.variance_epsilon)
    fileio_utils.save_int(rms_inv, 1 << 16, 'rms_inv_temp.bin')
    # print(rms_inv.shape)

    try:
        run_native_or_die(
            [
                "./rmsnorm",
                args.which,
                args.input_file,
                str(args.seq_len),
                str(embed_dim),
                workdir,
                layer_prefix,
                args.output_file,
            ],
            cwd=SCRIPT_DIR,
            name="rmsnorm",
        )
    finally:
        if os.path.isfile("rms_inv_temp.bin"):
            os.remove("rms_inv_temp.bin")
