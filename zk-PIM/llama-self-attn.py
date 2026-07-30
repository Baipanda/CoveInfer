import os, sys
import argparse
import torch
import numpy as np
import math

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

parser = argparse.ArgumentParser(description='LLaMa-2 Self-Attention')
parser.add_argument('model_size', type=int, choices = [7, 13], help='The size of the model to use. Default is 13')
parser.add_argument('layer', type=int, help='The layer to use for self-attn')
parser.add_argument('seq_len', type=int, help='The sequence length to use for self-attn')
parser.add_argument('--input_file', required = True, type=str, help='The input file to use for self-attn')
parser.add_argument('--output_file', default = 'llama-self-attn-output.bin', type=str, help='The output file to use for self-attn')
parser.add_argument('--model_path', type=str, default=None, help='Local path to a LLaMA-2 HF-format folder (overrides HF model id).')
parser.add_argument('--cache_dir', type=str, default='./model-storage', help='Transformers cache dir (used when loading from cache).')

from transformers import AutoTokenizer, AutoModelForCausalLM
from fileio_utils import *
from model_load_utils import resolve_model_ref, load_tokenizer_and_model

VALUE_LOGSF = 16
ACCU_LOGSF = 20

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _attn_head_layout(layer, model):
    """num_heads/head_dim lived on LlamaAttention in older transformers; newer releases use config."""
    attn = layer.self_attn
    cfg = model.config
    num_heads = getattr(attn, "num_heads", None)
    if num_heads is None:
        num_heads = getattr(cfg, "num_attention_heads", None)
    head_dim = getattr(attn, "head_dim", None)
    if head_dim is None:
        head_dim = getattr(cfg, "head_dim", None)
    if head_dim is None:
        hidden = getattr(cfg, "hidden_size", attn.q_proj.in_features)
        head_dim = hidden // int(num_heads)
    if num_heads is None or head_dim is None:
        raise AttributeError("Could not resolve attention num_heads/head_dim from model")
    return int(num_heads), int(head_dim)


def _rotary_cos_sin(model, seq_len: int, embed_dim: int, device: int = 0):
    """RoPE cos/sin: on LlamaModel in newer transformers, on LlamaAttention in older ones."""
    dummy = torch.randn(1, seq_len, embed_dim, device=device)
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

    rotary = getattr(model.model, "rotary_emb", None)
    if rotary is not None:
        rotary.to(device)
        try:
            return rotary(dummy, position_ids=position_ids)
        except TypeError:
            return rotary(dummy, position_ids)

    attn = model.model.layers[0].self_attn
    rotary = getattr(attn, "rotary_emb", None)
    if rotary is None:
        raise AttributeError("No rotary_emb on model.model or layer.self_attn")
    rotary.to(device)
    try:
        return rotary(dummy, position_ids)
    except TypeError:
        return rotary(dummy, seq_len)


if __name__ == '__main__':
    compilation_error = os.system('make self-attn')
    if compilation_error:
        print("Error compiling self-attn")
        exit(1)
    args = parser.parse_args()
    model_ref = resolve_model_ref(args.model_size, args.model_path)
    _, model = load_tokenizer_and_model(model_ref, cache_dir=args.cache_dir, local_files_only=True)
    layer = model.model.layers[args.layer]
    embed_dim = layer.self_attn.q_proj.in_features

    workdir = f'./zkllm-workdir/Llama-2-{args.model_size}b'
    layer_prefix = f'layer-{args.layer}'
    rc = os.system(f'./self-attn linear {args.input_file} {args.seq_len} {embed_dim} {workdir} {layer_prefix} {args.output_file}')
    if rc != 0:
        print(
            "self-attn linear failed.\n"
            "Likely root cause: commitment/weight mismatch (e.g., verifyWeightClaim failure), "
            "or workdir files generated from a different model/scaling.\n"
            "Please regenerate and align artifacts for the same model path:\n"
            "  python llama-ppgen.py <model_size> --model_path <same_model_path>\n"
            "  python llama-commit.py <model_size> 16 --model_path <same_model_path>\n"
            f"workdir: {workdir}, layer: {args.layer}\n"
        )
        sys.exit(1)

    missing = [p for p in ['temp_Q.bin', 'temp_K.bin', 'temp_V.bin'] if not os.path.isfile(p)]
    if missing:
        print(f"self-attn linear did not produce expected temp files: {missing}")
        sys.exit(1)

    Q, K, V = load_int('temp_Q.bin').reshape(args.seq_len, embed_dim) / (1 << 16), load_int('temp_K.bin').reshape(args.seq_len, embed_dim) / (1 << 16), load_int('temp_V.bin').reshape(args.seq_len, embed_dim) / (1 << 16)

    num_heads, head_dim = _attn_head_layout(layer, model)
    Q = Q.view(args.seq_len, num_heads, head_dim).transpose(0, 1)
    K = K.view(args.seq_len, num_heads, head_dim).transpose(0, 1)
    V = V.view(args.seq_len, num_heads, head_dim).transpose(0, 1)

    cos, sin = _rotary_cos_sin(model, args.seq_len, embed_dim, device=0)
    # cos/sin: [batch, seq_len, head_dim]; Q/K: [num_heads, seq_len, head_dim]
    if cos.dim() == 3 and cos.shape[0] == 1:
        cos = cos.squeeze(0).unsqueeze(0)
        sin = sin.squeeze(0).unsqueeze(0)

    Q, K = Q * cos + rotate_half(Q) * sin, K * cos + rotate_half(K) * sin
    Q, K = Q.to(torch.float64), K.to(torch.float64)
    
    A_ = Q @ K.transpose(-2, -1)
    A = to_int64(A_, VALUE_LOGSF)

    # an upper triangular mask for perplexity
    mask = torch.triu(torch.ones(args.seq_len, args.seq_len, device = 0, dtype = bool), diagonal = 1)

    A -= torch.max(A * ~mask, dim = -1, keepdim = True).values 

    shift = math.sqrt(head_dim) * torch.log((torch.exp((to_float(A, ACCU_LOGSF) / math.sqrt(head_dim))) * ~mask).sum(axis = -1, keepdim = True))
    shift = to_int64(shift, ACCU_LOGSF)
    A -= shift
    attn_output = (torch.exp(to_float(A, ACCU_LOGSF, torch.float64) / math.sqrt(head_dim)).float()) * ~mask

    attn_output = attn_output @ V
    attn_output = fromto_int64(attn_output, VALUE_LOGSF)

    attn_output = attn_output.transpose(0, 1).contiguous()
    attn_output = attn_output.view(args.seq_len, embed_dim)
    attn_output = attn_output.transpose(0, 1).reshape(args.seq_len, embed_dim)
    save_int(attn_output, 1 << 16, 'temp_attn_out.bin') 
    os.system(f'./self-attn attn {args.input_file} {args.seq_len} {embed_dim} {workdir} {layer_prefix} {args.output_file}')
    os.system('rm ./temp*.bin')

