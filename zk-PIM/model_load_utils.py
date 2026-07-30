import os
from typing import Optional, Tuple

from transformers import AutoModelForCausalLM, AutoTokenizer


def resolve_model_ref(model_size: int, model_path: Optional[str]) -> str:
    """
    Returns either a local directory path (preferred) or the HF model id.
    This repo originally hard-coded HF ids + local cache; we keep that as fallback.
    """
    if model_path:
        return model_path
    return f"meta-llama/Llama-2-{model_size}b-hf"


def load_tokenizer_and_model(
    model_ref: str,
    *,
    cache_dir: str = "./model-storage",
    local_files_only: bool = True,
) -> Tuple[object, object]:
    """
    Loads tokenizer + model from either a local directory or HF cache.
    If model_ref is a local path, we force local_files_only to avoid any network.
    """
    is_local_dir = os.path.isdir(model_ref)
    if is_local_dir:
        local_files_only = True

    tokenizer = AutoTokenizer.from_pretrained(
        model_ref,
        local_files_only=local_files_only,
        cache_dir=cache_dir,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_ref,
        local_files_only=local_files_only,
        cache_dir=cache_dir,
    )
    return tokenizer, model

