import os, sys
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

parser = argparse.ArgumentParser(description='LLaMa-2 PPGen')
parser.add_argument('model_size', type=int, choices = [7, 13], help='The size of the model to use. Default is 13')
parser.add_argument('--log_off_factor', type=int, default=5, help='The log offset factor to use. Default is 5')
parser.add_argument('--model_path', type=str, default=None, help='Local path to a LLaMA-2 HF-format folder (overrides HF model id).')
parser.add_argument('--cache_dir', type=str, default='./model-storage', help='Transformers cache dir (used when loading from cache).')

from transformers import AutoTokenizer, AutoModelForCausalLM
from model_load_utils import resolve_model_ref, load_tokenizer_and_model

if __name__ == '__main__':
    compilation_error = os.system('make ppgen')
    if compilation_error:
        print("Error compiling ppgen")
        exit(1)
    args = parser.parse_args()
    model_ref = resolve_model_ref(args.model_size, args.model_path)
    tokenizer, model = load_tokenizer_and_model(model_ref, cache_dir=args.cache_dir, local_files_only=True)

    os.makedirs(f"./zkllm-workdir/Llama-2-{args.model_size}b", exist_ok = True)

    for (i, w) in model.model.layers[0].named_parameters():
        if len(w.shape) == 2:
            pp_size = w.shape[0]
            pp_size <<= args.log_off_factor
        elif len(w.shape) == 1:
            (pp_size,) = w.shape
        else:
            raise ValueError(f"Unexpected shape {w.shape} for parameter {i}")
        
        os.system(f'./ppgen {pp_size} ./zkllm-workdir/Llama-2-{args.model_size}b/{i}-pp.bin')