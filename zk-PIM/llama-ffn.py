import os, sys
import argparse
import subprocess
import torch
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

from preflight_native import assert_input_int32_matrix, ffn_weight_paths, run_native_or_die

parser = argparse.ArgumentParser(description='LLaMa-2 Self-Attention')
parser.add_argument('model_size', type=int, choices=[7, 13], help='The size of the model to use. Default is 13')
parser.add_argument('layer', type=int, help='The layer to use for ffn')
parser.add_argument('seq_len', type=int, help='The sequence length to use for ffn')
parser.add_argument('--input_file', required=True, type=str, help='The input file to use for ffn')
parser.add_argument('--output_file', default='llama-ffn-output.bin', type=str, help='The output file to use for ffn')
parser.add_argument('--model_path', type=str, default=None, help='Local path to a LLaMA-2 HF-format folder (overrides HF model id).')
parser.add_argument('--cache_dir', type=str, default='./model-storage', help='Transformers cache dir (used when loading from cache).')

from transformers import AutoModelForCausalLM
import fileio_utils
from model_load_utils import resolve_model_ref, load_tokenizer_and_model


def prepare_swiglu(in_range_num_bit=10, in_prec_num_bit=12, out_prec_num_bit=16):
    Xs = torch.arange(
        -(1 << (in_range_num_bit - 1)),
        1 << (in_range_num_bit - 1),
        step=1 / (1 << in_prec_num_bit),
        device=0,
    )
    Ys = Xs * torch.sigmoid(Xs)
    fileio_utils.save_int(Ys, out_prec_num_bit, 'swiglu-table.bin')


if __name__ == '__main__':
    args = parser.parse_args()
    if not str(args.input_file).strip() or not str(args.output_file).strip():
        print(
            "ERROR: --input_file / --output_file look empty. "
            "Set shell variables or use literals, e.g.\n"
            "  python llama-ffn.py 7 0 2048 --input_file /tmp/ffn_in.bin --output_file /tmp/out.bin --model_path ...",
            file=sys.stderr,
        )
        sys.exit(2)

    compilation_error = subprocess.call(["make", "ffn"], cwd=SCRIPT_DIR)
    if compilation_error:
        print("Error compiling ffn")
        sys.exit(1)

    prepare_swiglu()

    model_ref = resolve_model_ref(args.model_size, args.model_path)
    _, model = load_tokenizer_and_model(model_ref, cache_dir=args.cache_dir, local_files_only=True)
    layer = model.model.layers[0]
    embed_dim, hidden_dim = layer.mlp.up_proj.in_features, layer.mlp.up_proj.out_features

    workdir = f'./zkllm-workdir/Llama-2-{args.model_size}b'
    layer_prefix = f'layer-{args.layer}'

    miss = [p for p in ffn_weight_paths(workdir, layer_prefix) if not os.path.isfile(p)]
    if miss:
        print(
            "ERROR: native ffn needs these files (missing):\n  "
            + "\n  ".join(miss[:12])
            + ("\n  ..." if len(miss) > 12 else "")
            + "\nRun llama-ppgen.py then llama-commit.py for this model.",
            file=sys.stderr,
        )
        sys.exit(2)

    if not os.path.isfile(args.input_file):
        fileio_utils.save_int(torch.randn(args.seq_len, embed_dim, device=0), 1 << 16, args.input_file)
    assert_input_int32_matrix(args.input_file, args.seq_len, embed_dim, label="ffn input")

    try:
        run_native_or_die(
            [
                "./ffn",
                args.input_file,
                str(args.seq_len),
                str(embed_dim),
                str(hidden_dim),
                workdir,
                layer_prefix,
                args.output_file,
            ],
            cwd=SCRIPT_DIR,
            name="ffn",
        )
    finally:
        if os.path.isfile('swiglu-table.bin'):
            os.remove('swiglu-table.bin')
