import argparse
import contextlib
import gc
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Optional

import gradio as gr
import pandas as pd
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer

from cove_paths import env_str, resolve_results_dir, EVAL_RESULTS_DIR, sage_enroll_iters, sage_enroll_runs, sage_profile_path, sage_verify_threads
from cove_ui_theme import build_gradio_css, current_theme_name

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RESULTS_DIR = EVAL_RESULTS_DIR
DEFAULT_CACHE_DIR = os.path.join(REPO_ROOT, "model-storage")
SAGE_DIR = os.path.join(REPO_ROOT, "LSAGE")
SAGE_PROFILE = sage_profile_path()
EVAL_CONFIGS = ["CSV+4090", "H20", "CSV+5090"]
FORCE_MATH_SDP_ENV = "COVE_FORCE_MATH_SDP"
SAGE_RUNTIME_RE = re.compile(r"Runtime:\s*([0-9]+(?:\.[0-9]+)?)\s*s")
UI_CSS = build_gradio_css(max_width="1180px", with_chat_extras=False)

def _force_math_sdp() -> None:
    try:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        return


def _math_sdp_env_enabled() -> bool:
    return env_str(
        FORCE_MATH_SDP_ENV,
        "CVEE_FORCE_MATH_SDP",
        "CVINF_FORCE_MATH_SDP",
        "ZKLLM_FORCE_MATH_SDP",
    ) == "1"


def _should_force_math_sdp(*, device: str, config_name: str) -> bool:
    if not str(device).startswith("cuda"):
        return False
    if _math_sdp_env_enabled():
        return True
    if str(config_name).strip().upper() == "H20":
        return True
    # Auto-detect by device name.
    try:
        if torch.cuda.is_available():
            idx = torch.device(device).index or 0
            name = torch.cuda.get_device_name(idx)
            if "H20" in str(name).upper():
                return True
    except Exception:
        pass
    return False


def _maybe_force_math_sdp(*, device: str, config_name: str) -> None:
    """
    Work around GPU-kernel-level crashes observed on some Hopper setups
    (e.g. H20) with Flash/MemEfficient SDPA.

    Prefer math-only SDPA for stability on H20/Hopper setups.

    Triggers when either:
    - config_name == 'H20' (explicit), OR
    - current CUDA device name contains 'H20' (auto-detect), OR
    - env COVE_FORCE_MATH_SDP=1 (manual override)
    """
    if not _should_force_math_sdp(device=device, config_name=config_name):
        return
    _force_math_sdp()


def _now_ts() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _run_cmd(cmd: list[str], timeout_sec: int = 10) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, text=True, capture_output=True, check=False, timeout=timeout_sec)
        out = (p.stdout or "") + ("\n" + p.stderr if p.stderr else "")
        return p.returncode, out.strip()
    except FileNotFoundError as e:
        return 127, f"Command not found: {e.filename}"


def _resolve_nvcc() -> Optional[str]:
    candidates = [
        shutil.which("nvcc"),
        "/usr/local/cuda/bin/nvcc",
        os.path.join(sys.prefix, "bin", "nvcc"),
    ]
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    if conda_prefix:
        candidates.append(os.path.join(conda_prefix, "bin", "nvcc"))
    for path in candidates:
        if path and os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    return None


def _sage_env() -> dict[str, str]:
    env = os.environ.copy()
    nvcc = _resolve_nvcc()
    if nvcc:
        env["PATH"] = f"{os.path.dirname(nvcc)}:{env.get('PATH', '')}"
        env["NVCC"] = nvcc
    return env


def _run_sage_attestation_cmd(cmd: list[str], timeout_sec: int = 1800) -> tuple[int, str]:
    if not os.path.isdir(SAGE_DIR):
        return 1, f"SAGE directory not found: {SAGE_DIR}"
    if _resolve_nvcc() is None:
        return 1, "nvcc not found. Expected CUDA compiler at /usr/local/cuda/bin/nvcc or in PATH."
    try:
        p = subprocess.run(
            cmd,
            cwd=SAGE_DIR,
            env=_sage_env(),
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout_sec,
        )
        out = (p.stdout or "") + ("\n" + p.stderr if p.stderr else "")
        return p.returncode, out.strip()
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or "") + ("\n" + e.stderr if e.stderr else "")
        return 124, f"Timeout after {timeout_sec}s\n{out}"


def _parse_sage_attestation(code: int, out: str) -> dict[str, Any]:
    runtime = None
    m = SAGE_RUNTIME_RE.search(out or "")
    if m:
        runtime = float(m.group(1))
    checksum_ok = "checksum_ok=True" in out
    time_ok = "time_ok=True" in out
    ok = (code == 0) and ("verification SUCCEED" in out)
    return {
        "ok": ok,
        "exit_code": code,
        "runtime_s": runtime,
        "checksum_ok": checksum_ok,
        "time_ok": time_ok,
        "profile": SAGE_PROFILE,
        "output_tail": (out or "")[-4000:],
    }


def run_sage_enroll(*, cap: str, runs: int, iters: int, data_size: int, threshold_sigma_k: float) -> tuple[dict[str, Any], str]:
    cmd = [
        sys.executable,
        os.path.join(SAGE_DIR, "run_attestation.py"),
        "enroll",
        "--profile",
        SAGE_PROFILE,
        "--gpu-model",
        "H20",
        "--cap",
        str(cap),
        "--runs",
        str(int(runs)),
        "--iters",
        str(int(iters)),
        "--data-size",
        str(int(data_size)),
        "--threshold-sigma-k",
        str(float(threshold_sigma_k)),
        "--verify-threads",
        str(sage_verify_threads()),
    ]
    code, out = _run_sage_attestation_cmd(cmd)
    result = {"ok": code == 0, "exit_code": code, "profile": SAGE_PROFILE, "output_tail": out[-4000:]}
    return result, out


def run_sage_attest(
    *,
    auto_enroll: bool,
    cap: str,
    runs: int,
    iters: int,
    data_size: int,
    threshold_sigma_k: float,
) -> tuple[dict[str, Any], str]:
    prefix = ""
    if auto_enroll and not os.path.isfile(SAGE_PROFILE):
        enroll_result, enroll_out = run_sage_enroll(
            cap=cap,
            runs=runs,
            iters=iters,
            data_size=data_size,
            threshold_sigma_k=threshold_sigma_k,
        )
        prefix = "[auto-enroll]\n" + enroll_out + "\n\n"
        if not enroll_result.get("ok"):
            return enroll_result, prefix

    cmd = [
        sys.executable,
        os.path.join(SAGE_DIR, "run_attestation.py"),
        "attest",
        "--profile",
        SAGE_PROFILE,
        "--verify-threads",
        str(sage_verify_threads()),
    ]
    code, out = _run_sage_attestation_cmd(cmd)
    result = _parse_sage_attestation(code, out)
    return result, prefix + out


def _nvidia_smi_query() -> dict[str, Any]:
    """
    Lightweight GPU metrics without extra deps.
    Returns empty dict if nvidia-smi unavailable.
    """
    fields = [
        "name",
        "driver_version",
        "utilization.gpu",
        "utilization.memory",
        "memory.used",
        "memory.total",
        "temperature.gpu",
        "power.draw",
        "clocks.sm",
    ]
    code, out = _run_cmd(
        [
            "nvidia-smi",
            f"--query-gpu={','.join(fields)}",
            "--format=csv,noheader,nounits",
        ],
        timeout_sec=10,
    )
    if code != 0 or not out:
        return {}
    parts = [p.strip() for p in out.split(",")]
    if len(parts) != len(fields):
        return {}
    d: dict[str, Any] = {"source": "nvidia-smi"}
    for k, v in zip(fields, parts):
        d[k] = v
    for k in ["utilization.gpu", "utilization.memory", "memory.used", "memory.total", "temperature.gpu", "power.draw", "clocks.sm"]:
        if k in d:
            try:
                d[k] = float(d[k])
            except Exception:
                pass
    return d


def _pad_to_length(tokenizer: AutoTokenizer, target_tokens: int) -> str:
    # Deterministic filler prompt: short, repeatable, tokenizer-agnostic.
    base = "Hello"
    s = base
    for _ in range(20000):
        n = len(tokenizer(s, add_special_tokens=False)["input_ids"])
        if n >= target_tokens:
            break
        s += " " + base
    return s


@dataclass
class Scenario:
    config: str
    input_tokens: int
    output_tokens: int
    batch: int


@dataclass
class RunMetrics:
    config: str
    model_ref: str
    dtype: str
    device: str
    input_tokens: int
    output_tokens: int
    batch: int
    ttft_ms: float
    total_ms: float
    tokens_per_s: float
    transfer_ms: float
    gpu: dict[str, Any]
    started_at: str


def _low_ram_load_enabled() -> bool:
    return env_str("COVE_LOW_RAM_LOAD", "CVEE_LOW_RAM_LOAD") == "1"


def _cuda_device_name(device: str) -> str:
    try:
        if not device.startswith("cuda") or not torch.cuda.is_available():
            return ""
        idx = torch.device(device).index or 0
        return str(torch.cuda.get_device_name(idx))
    except Exception:
        return ""


def _is_h20_device(device: str) -> bool:
    return "H20" in _cuda_device_name(device).upper()


def _is_qwen_family(model_ref: str) -> bool:
    try:
        cfg = AutoConfig.from_pretrained(model_ref, local_files_only=True)
        return getattr(cfg, "model_type", "") in ("qwen2", "qwen3")
    except Exception:
        return "qwen" in os.path.basename(model_ref).lower()


def _load_causal_lm_low_ram(model_ref: str, *, torch_dtype: torch.dtype, device: str) -> AutoModelForCausalLM:
    """
    Load sharded safetensors directly onto GPU to avoid CPU RAM spikes
  (needed on ~14GB-RAM hosts when loading Qwen2.5-7B).
    """
    from safetensors.torch import load_file

    dev = torch.device(device)
    config = AutoConfig.from_pretrained(model_ref, local_files_only=True)
    with torch.device(dev):
        model = AutoModelForCausalLM.from_config(config, dtype=torch_dtype)
    shards = sorted(glob.glob(os.path.join(model_ref, "model-*.safetensors")))
    if not shards:
        raise FileNotFoundError(
            f"COVE_LOW_RAM_LOAD=1 but no model-*.safetensors under {model_ref}"
        )
    for shard in shards:
        state = load_file(shard, device=str(dev))
        if torch_dtype == torch.float32:
            state = {k: v.float() for k, v in state.items()}
        model.load_state_dict(state, strict=False)
        del state
        gc.collect()
    model.eval()
    return model


def load_model(model_ref: str, *, dtype: str, device: str) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "device requests CUDA but torch.cuda.is_available() is False.\n"
            "Fix: install a CUDA-enabled PyTorch build, ensure NVIDIA driver works (nvidia-smi), and rerun.\n"
            f"device={device}"
        )

    # Defensive: force math SDPA on H20 even if caller forgot to.
    # We don't have config_name here, so rely on env var or device name.
    try:
        if _math_sdp_env_enabled() and device.startswith("cuda"):
            _force_math_sdp()
        elif device.startswith("cuda") and _is_h20_device(device):
            _force_math_sdp()
    except Exception:
        pass

    qwen_model = _is_qwen_family(model_ref)
    h20_device = device.startswith("cuda") and _is_h20_device(device)

    # H20 stability: bf16 can trigger SIGFPE on Llama; Qwen fp16/bf16 also SIGFPE on H20.
    if dtype == "bf16" and h20_device and not qwen_model:
        dtype = "fp16"

    torch_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}.get(dtype, torch.float16)
    if qwen_model and h20_device and torch_dtype in (torch.float16, torch.bfloat16):
        print(
            "[cove] Qwen on H20: fp16/bf16 can SIGFPE; using fp32 for inference.",
            file=sys.stderr,
        )
        torch_dtype = torch.float32
    if torch_dtype == torch.bfloat16 and device.startswith("cuda"):
        # Most modern datacenter GPUs support bf16; guard anyway for clearer errors.
        try:
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError("bf16 is not supported by this GPU / CUDA stack; use --dtype fp16 instead.")
        except AttributeError:
            # Older torch may not have is_bf16_supported; proceed.
            pass

    if not (os.path.isdir(model_ref) or os.path.isfile(model_ref)):
        # evaluation.py forces local_files_only=True; make the failure mode explicit.
        raise FileNotFoundError(
            "model_ref must be a LOCAL directory (or an existing local HF cache entry).\n"
            f"Got: {model_ref}\n"
            "Fix: pass --model_ref /path/to/local/model (containing config.json + weights + tokenizer files)."
        )

    tok = AutoTokenizer.from_pretrained(model_ref, local_files_only=True, cache_dir=DEFAULT_CACHE_DIR)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    if _low_ram_load_enabled():
        model = _load_causal_lm_low_ram(model_ref, torch_dtype=torch_dtype, device=device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_ref,
            local_files_only=True,
            cache_dir=DEFAULT_CACHE_DIR,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        )
        model.eval()
        model.to(torch.device(device))
    return tok, model


def _measure_transfer_ms(input_ids_cpu: torch.Tensor, attention_mask_cpu: torch.Tensor, device: str) -> float:
    if not device.startswith("cuda"):
        return 0.0
    torch.cuda.synchronize()
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    _ = input_ids_cpu.to(device, non_blocking=True)
    _ = attention_mask_cpu.to(device, non_blocking=True)
    t1.record()
    torch.cuda.synchronize()
    return float(t0.elapsed_time(t1))


def run_one(
    tok: AutoTokenizer,
    model: AutoModelForCausalLM,
    scenario: Scenario,
    *,
    device: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> RunMetrics:
    prompt = _pad_to_length(tok, scenario.input_tokens)
    prompts = [prompt for _ in range(max(1, int(scenario.batch)))]
    enc = tok(prompts, return_tensors="pt", padding=True, truncation=False, add_special_tokens=False)

    input_ids_cpu = enc["input_ids"].contiguous()
    attn_cpu = enc["attention_mask"].contiguous()
    transfer_ms = _measure_transfer_ms(input_ids_cpu, attn_cpu, device)

    input_ids = input_ids_cpu.to(device)
    attention_mask = attn_cpu.to(device)

    ttft_ms = float("nan")
    t_start = time.perf_counter()
    gpu_before = _nvidia_smi_query()

    # Prefer math-only SDPA on H20 to avoid kernel-level SIGFPE.
    use_math_sdp = _should_force_math_sdp(device=device, config_name=scenario.config)

    def _sdp_ctx():
        if not (use_math_sdp and device.startswith("cuda")):
            return contextlib.nullcontext()
        try:
            # NOTE: this API is deprecated in newer torch; still works on current stack.
            # Create a NEW context manager each time (it's not re-entrant).
            return torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)
        except Exception:
            return contextlib.nullcontext()

    # TTFT:
    # - For batch=1: use streamer to measure real "first token to client"
    # - For batch>1: TextStreamer doesn't support batching; use a prefill forward-pass time as a proxy.
    if int(scenario.batch) == 1:
        streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=int(max_new_tokens),
            do_sample=(temperature > 0.0),
            temperature=float(temperature),
            top_p=float(top_p),
            streamer=streamer,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
            use_cache=True,
        )

        # generate() blocks; run in background so we can observe first token via streamer.
        import threading

        def _bg():
            with torch.inference_mode():
                with _sdp_ctx():
                    model.generate(**gen_kwargs)

        thread = threading.Thread(target=_bg, daemon=True)
        thread.start()

        for _chunk in streamer:
            ttft_ms = (time.perf_counter() - t_start) * 1000.0
            break
        thread.join()
    else:
        # Prefill proxy: one forward pass with cache to approximate first-token latency.
        if device.startswith("cuda"):
            torch.cuda.synchronize()
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            with torch.inference_mode():
                with _sdp_ctx():
                    _ = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
            e1.record()
            torch.cuda.synchronize()
            ttft_ms = float(e0.elapsed_time(e1))
        else:
            t0 = time.perf_counter()
            with torch.inference_mode():
                _ = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
            ttft_ms = (time.perf_counter() - t0) * 1000.0

        gen_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=int(max_new_tokens),
            do_sample=(temperature > 0.0),
            temperature=float(temperature),
            top_p=float(top_p),
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
            use_cache=True,
        )
        with torch.inference_mode():
            with _sdp_ctx():
                _ = model.generate(**gen_kwargs)

    total_ms = (time.perf_counter() - t_start) * 1000.0
    gpu_after = _nvidia_smi_query()

    # Approx tokens/s: count generated tokens by re-tokenizing generated text per sample is expensive.
    # We use max_new_tokens as the intended output length for consistent experiment comparisons.
    new_tokens = int(max_new_tokens) * int(scenario.batch)
    tokens_per_s = (new_tokens / (total_ms / 1000.0)) if total_ms > 0 else 0.0

    gpu = {"before": gpu_before, "after": gpu_after}
    return RunMetrics(
        config=scenario.config,
        model_ref=str(getattr(tok, "name_or_path", "")) or "unknown",
        dtype=str(getattr(model, "dtype", "")),
        device=device,
        input_tokens=int(scenario.input_tokens),
        output_tokens=int(max_new_tokens),
        batch=int(scenario.batch),
        ttft_ms=float(ttft_ms),
        total_ms=float(total_ms),
        tokens_per_s=float(tokens_per_s),
        transfer_ms=float(transfer_ms),
        gpu=gpu,
        started_at=_now_ts(),
    )


def default_scenarios(config_name: str) -> list[Scenario]:
    # Designed for a single-GPU node; keep runtime bounded.
    inputs = [128, 512, 1024]
    outputs = [128]
    batches = [1, 2, 4]
    out: list[Scenario] = []
    for i in inputs:
        for o in outputs:
            for b in batches:
                out.append(Scenario(config=config_name, input_tokens=i, output_tokens=o, batch=b))
    return out


def _results_paths(results_dir: str, run_name: str) -> tuple[str, str]:
    os.makedirs(results_dir, exist_ok=True)
    jsonl = os.path.join(results_dir, f"{run_name}.jsonl")
    csv = os.path.join(results_dir, f"{run_name}.csv")
    return jsonl, csv


def _result_file_choices(results_dir: str) -> list[str]:
    if not os.path.isdir(results_dir):
        return []
    out: list[tuple[float, str]] = []
    for root, _dirs, files in os.walk(results_dir):
        for name in files:
            if not name.endswith((".csv", ".jsonl")):
                continue
            path = os.path.join(root, name)
            rel = os.path.relpath(path, results_dir)
            stem, ext = os.path.splitext(name)
            parts = stem.rsplit("-", 2)
            config = parts[0] if parts else stem
            ts_label = ""
            if len(parts) >= 3:
                try:
                    ts = datetime.strptime(f"{parts[-2]}-{parts[-1]}", "%Y%m%d-%H%M%S")
                    ts_label = ts.strftime("%Y-%m-%d %H:%M:%S")
                except ValueError:
                    ts_label = f"{parts[-2]}-{parts[-1]}"
            model_label = os.path.dirname(rel)
            label_bits = [config]
            if model_label and model_label != ".":
                label_bits.append(model_label)
            if ts_label:
                label_bits.append(ts_label)
            label_bits.append(ext.lstrip(".").upper())
            label_bits.append(rel)
            out.append((os.path.getmtime(path), " | ".join(label_bits)))
    return [label for _mtime, label in sorted(out, key=lambda item: item[0], reverse=True)]


def _result_label_to_path(selection: str, results_dir: str) -> str:
    # Labels end with the relative path under results_dir.
    file_name = str(selection).split(" | ")[-1].strip()
    return os.path.join(results_dir, file_name)


def run_suite(
    *,
    model_ref: str,
    config_name: str,
    device: str,
    dtype: str,
    results_dir: str,
    warmup: int,
    repeat: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    save_format: str = "jsonl",
    gpu_verify: bool = False,
    sage_cap: str = "90",
    sage_runs: int = 3,
    sage_iters: int = 10000,
    sage_data_size: int = 1 << 20,
    sage_threshold_sigma_k: float = 2.0,
) -> tuple[pd.DataFrame, str]:
    """
    save_format: "jsonl" | "csv" | "both" — which artifact files to write under results_dir.
    """
    fmt = str(save_format).strip().lower()
    if fmt not in ("jsonl", "csv", "both"):
        fmt = "jsonl"
    want_jsonl = fmt in ("jsonl", "both")
    want_csv = fmt in ("csv", "both")

    run_name = f"{config_name}-{_now_ts()}"
    effective_results_dir = resolve_results_dir(model_ref, results_dir)
    jsonl_path, csv_path = _results_paths(effective_results_dir, run_name)

    gpu_attestation: Optional[dict[str, Any]] = None
    if gpu_verify:
        gpu_attestation, attest_out = run_sage_attest(
            auto_enroll=True,
            cap=sage_cap,
            runs=sage_runs,
            iters=sage_iters,
            data_size=sage_data_size,
            threshold_sigma_k=sage_threshold_sigma_k,
        )
        if not gpu_attestation.get("ok"):
            raise RuntimeError(f"GPU verification failed; benchmark aborted.\n{attest_out[-3000:]}")

    _maybe_force_math_sdp(device=device, config_name=config_name)
    tok, model = load_model(model_ref, dtype=dtype, device=device)
    scenarios = default_scenarios(config_name)

    # Warmup: 1 short run to trigger lazy CUDA init, kernels, cache, etc.
    for _ in range(max(0, int(warmup))):
        _ = run_one(
            tok,
            model,
            Scenario(config=config_name, input_tokens=32, output_tokens=32, batch=1),
            device=device,
            max_new_tokens=16,
            temperature=temperature,
            top_p=top_p,
        )

    rows: list[dict[str, Any]] = []
    jsonl_f = open(jsonl_path, "w", encoding="utf-8") if want_jsonl else None
    try:
        for sc in scenarios:
            for _r in range(max(1, int(repeat))):
                m = run_one(
                    tok,
                    model,
                    sc,
                    device=device,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
                d = asdict(m)
                if gpu_attestation is not None:
                    d["gpu_attestation"] = gpu_attestation
                if jsonl_f is not None:
                    jsonl_f.write(json.dumps(d, ensure_ascii=False) + "\n")
                rows.append(d)
    finally:
        if jsonl_f is not None:
            jsonl_f.close()

    df = pd.json_normalize(rows)
    saved: list[str] = []
    if want_csv:
        df.to_csv(csv_path, index=False)
        saved.append(csv_path)
    if want_jsonl:
        saved.append(jsonl_path)
    note = "\n".join(f"Saved: {p}" for p in saved) if saved else "No files written (unexpected)."
    if gpu_attestation is not None:
        note = f"GPU verification: PASS (runtime={gpu_attestation.get('runtime_s')}s)\n{note}"
    return df, note


def _df_for_display(df: pd.DataFrame) -> pd.DataFrame:
    # Display a clean table similar to your screenshot; keep extended columns in raw export.
    cols = [
        "config",
        "input_tokens",
        "output_tokens",
        "batch",
        "ttft_ms",
        "tokens_per_s",
        "transfer_ms",
        "gpu.after.utilization.gpu",
        "gpu.after.memory.used",
    ]
    keep = [c for c in cols if c in df.columns]
    out = df[keep].copy()
    rename = {
        "config": "Config",
        "input_tokens": "input(tokens)",
        "output_tokens": "output(tokens)",
        "batch": "batch",
        "ttft_ms": "TTFT(ms)",
        "tokens_per_s": "tokens/s(approx)",
        "transfer_ms": "H2D(ms)",
        "gpu.after.utilization.gpu": "GPU util(%)",
        "gpu.after.memory.used": "mem used(MiB)",
    }
    out = out.rename(columns=rename)
    for c in ["TTFT(ms)", "tokens/s(approx)", "H2D(ms)", "GPU util(%)", "mem used(MiB)"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").round(2)
    return out


def build_ui():
    initial_choices = _result_file_choices(DEFAULT_RESULTS_DIR)
    with gr.Blocks(title="Cove Evaluation") as demo:
        gr.HTML(
            """
            <div class="hero">
              <h1>Cove Evaluation Dashboard</h1>
              <p>Run H20 benchmarks, compare CSV+4090 / CSV+5090 results, and open saved runs from eval-results (dropdown loads on change).</p>
            </div>
            """
        )

        with gr.Group(elem_classes=["section-card"]):
            gr.Markdown("### Run Configuration")
            with gr.Row():
                with gr.Column(scale=2):
                    model_ref = gr.Textbox(label="Model path / ID (local directory recommended)", value="/home/data/models/Llama-2-7b-hf")
                    config_name = gr.Dropdown(label="Config name", choices=EVAL_CONFIGS, value="CSV+4090")
                    results_dir = gr.Textbox(label="results_dir", value=DEFAULT_RESULTS_DIR)
                with gr.Column(scale=1):
                    device = gr.Textbox(label="device", value="cuda:0")
                    dtype = gr.Dropdown(label="dtype", choices=["fp16", "bf16", "fp32"], value="fp16")
                    max_new_tokens = gr.Slider(label="max_new_tokens (output length)", minimum=16, maximum=512, value=128, step=8)
            with gr.Row():
                warmup = gr.Slider(label="warmup runs", minimum=0, maximum=3, value=1, step=1)
                repeat = gr.Slider(label="repeat per scenario", minimum=1, maximum=5, value=1, step=1)
                temperature = gr.Slider(label="temperature", minimum=0.0, maximum=1.5, value=0.0, step=0.05)
                top_p = gr.Slider(label="top_p", minimum=0.1, maximum=1.0, value=0.9, step=0.05)
            save_format = gr.Radio(
                label="Result file format",
                choices=[
                    ("JSONL only (default)", "jsonl"),
                    ("CSV only", "csv"),
                    ("JSONL + CSV", "both"),
                ],
                value="jsonl",
            )
            run_btn = gr.Button(">>> RUN BENCHMARK <<<", variant="primary", elem_classes=["primary-btn"])
            note = gr.Textbox(label="Saved files", lines=3)

        with gr.Group(elem_classes=["section-card"]):
            gr.Markdown("### GPU Verification (SAGE)")
            gr.Markdown("SAGE attestation validates the GPU before running evaluation. Use enroll once to create a baseline profile, then attest before benchmark runs.")
            with gr.Row():
                sage_cap = gr.Textbox(label="CUDA SM capability", value="90")
                sage_runs = gr.Slider(label="enroll baseline runs", minimum=1, maximum=30, value=30, step=1)
                sage_iters = gr.Slider(label="attestation iters", minimum=1000, maximum=100000, value=10000, step=1000)
                sage_data_size = gr.Number(label="data_size", value=1 << 20, precision=0)
                sage_threshold_sigma_k = gr.Slider(label="threshold sigma k", minimum=0.5, maximum=5.0, value=2.0, step=0.5)
            gpu_verify = gr.Checkbox(label="Run GPU verification before benchmark", value=True)
            sage_enroll_btn = gr.Button(">>> ENROLL GPU BASELINE <<<", variant="secondary", elem_classes=["full-width-btn"])
            sage_attest_btn = gr.Button(">>> RUN GPU ATTESTATION <<<", variant="secondary", elem_classes=["full-width-btn"])
            sage_status = gr.Textbox(label="GPU verification status", lines=10)

        with gr.Group(elem_classes=["section-card"]):
            gr.Markdown("### Results Table (sortable/copyable; full columns are in the selected export format)")
            table = gr.Dataframe(interactive=False, wrap=True)

        with gr.Group(elem_classes=["section-card"]):
            gr.Markdown("### Load Existing Results (from eval-results; changing the dropdown loads immediately)")
            result_file = gr.Dropdown(
                label="Result file",
                choices=initial_choices,
                value=(initial_choices[0] if initial_choices else None),
            )
            refresh_btn = gr.Button("Refresh file list")
            load_note = gr.Textbox(label="Load status", lines=2)

        state_df = gr.State(pd.DataFrame())

        def _run(
            model_ref,
            config_name,
            device,
            dtype,
            results_dir,
            warmup,
            repeat,
            max_new_tokens,
            temperature,
            top_p,
            save_format,
            gpu_verify,
            sage_cap,
            sage_runs,
            sage_iters,
            sage_data_size,
            sage_threshold_sigma_k,
        ):
            df, msg = run_suite(
                model_ref=model_ref,
                config_name=config_name,
                device=device,
                dtype=dtype,
                results_dir=results_dir,
                warmup=int(warmup),
                repeat=int(repeat),
                max_new_tokens=int(max_new_tokens),
                temperature=float(temperature),
                top_p=float(top_p),
                save_format=str(save_format or "jsonl"),
                gpu_verify=bool(gpu_verify),
                sage_cap=str(sage_cap or "90"),
                sage_runs=int(sage_runs),
                sage_iters=int(sage_iters),
                sage_data_size=int(sage_data_size),
                sage_threshold_sigma_k=float(sage_threshold_sigma_k),
            )
            disp = _df_for_display(df)
            choices = _result_file_choices(results_dir)
            return df, disp, msg, gr.update(choices=choices, value=choices[0] if choices else None)

        def _sage_enroll(sage_cap, sage_runs, sage_iters, sage_data_size, sage_threshold_sigma_k):
            result, out = run_sage_enroll(
                cap=str(sage_cap or "90"),
                runs=int(sage_runs),
                iters=int(sage_iters),
                data_size=int(sage_data_size),
                threshold_sigma_k=float(sage_threshold_sigma_k),
            )
            status = "PASS" if result.get("ok") else "FAIL"
            return f"Enroll {status}\n{out}"

        def _sage_attest(sage_cap, sage_runs, sage_iters, sage_data_size, sage_threshold_sigma_k):
            result, out = run_sage_attest(
                auto_enroll=True,
                cap=str(sage_cap or "90"),
                runs=int(sage_runs),
                iters=int(sage_iters),
                data_size=int(sage_data_size),
                threshold_sigma_k=float(sage_threshold_sigma_k),
            )
            status = "PASS" if result.get("ok") else "FAIL"
            return f"Attestation {status}\n{out}"

        def _preset(config_name: str):
            # Keep current behavior for existing configs; add a helpful default for H20.
            if str(config_name).strip().upper() == "H20":
                # Prefer fp16 on H20 for stability; bf16 can SIGFPE on some stacks.
                return gr.update(value="fp16")
            return gr.update()

        def _refresh_result_files(results_dir: str):
            choices = _result_file_choices(results_dir)
            return gr.update(choices=choices, value=choices[0] if choices else None)

        def _load_selected(selection: str, results_dir: str):
            if not selection:
                empty = pd.DataFrame()
                return empty, empty, "No file."
            path = selection if os.path.isabs(selection) else _result_label_to_path(selection, results_dir)
            try:
                if path.endswith(".csv"):
                    df_new = pd.read_csv(path)
                elif path.endswith(".jsonl"):
                    rows = []
                    with open(path, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            rows.append(json.loads(line))
                    df_new = pd.json_normalize(rows)
                else:
                    empty = pd.DataFrame()
                    return empty, empty, f"Unsupported: {path}"
            except Exception as e:
                empty = pd.DataFrame()
                return empty, empty, f"Load failed: {e}"

            configs = ", ".join(sorted(str(x) for x in df_new.get("config", pd.Series(dtype=str)).dropna().unique()))
            extra = f" configs={configs}" if configs else ""
            return df_new, _df_for_display(df_new), f"Loaded: {os.path.basename(path)} rows={len(df_new)}{extra}"

        run_btn.click(
            fn=_run,
            inputs=[
                model_ref,
                config_name,
                device,
                dtype,
                results_dir,
                warmup,
                repeat,
                max_new_tokens,
                temperature,
                top_p,
                save_format,
                gpu_verify,
                sage_cap,
                sage_runs,
                sage_iters,
                sage_data_size,
                sage_threshold_sigma_k,
            ],
            outputs=[state_df, table, note, result_file],
        )

        config_name.change(fn=_preset, inputs=[config_name], outputs=[dtype])

        sage_enroll_btn.click(
            fn=_sage_enroll,
            inputs=[sage_cap, sage_runs, sage_iters, sage_data_size, sage_threshold_sigma_k],
            outputs=[sage_status],
        )

        sage_attest_btn.click(
            fn=_sage_attest,
            inputs=[sage_cap, sage_runs, sage_iters, sage_data_size, sage_threshold_sigma_k],
            outputs=[sage_status],
        )

        refresh_btn.click(fn=_refresh_result_files, inputs=[results_dir], outputs=[result_file])

        result_file.change(
            fn=_load_selected,
            inputs=[result_file, results_dir],
            outputs=[state_df, table, load_note],
        )

    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ui", action="store_true", help="Launch dashboard")
    ap.add_argument("--model_ref", default="/home/data/models/Llama-2-7b-hf")
    ap.add_argument("--config", default="CSV+4090")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default=None, choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--results_dir", default=DEFAULT_RESULTS_DIR)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--gpu-verify", action="store_true", help="Run SAGE GPU attestation before benchmark")
    ap.add_argument("--sage-cap", default="90")
    ap.add_argument("--sage-runs", type=int, default=3)
    ap.add_argument("--sage-iters", type=int, default=10000)
    ap.add_argument("--sage-data-size", type=int, default=1 << 20)
    ap.add_argument("--sage-threshold-sigma-k", type=float, default=2.0)
    ap.add_argument(
        "--save-format",
        default="jsonl",
        choices=["jsonl", "csv", "both"],
        help="Which result files to write: jsonl, csv, or both.",
    )
    args = ap.parse_args()

    if args.dtype is None:
        # Prefer fp16 as a stable default across setups (incl. H20).
        args.dtype = "fp16"

    # UI can receive requests immediately after launch; force stable SDPA early
    # so the worker thread never sees flash/mem-efficient SDPA on H20.
    _maybe_force_math_sdp(device=args.device, config_name=args.config)

    if args.ui:
        print(f"* UI theme: {current_theme_name()} (set COVE_UI_THEME=warm_gray|ivory|…)")
        demo = build_ui()
        demo.launch(server_name="0.0.0.0", server_port=7861, share=False, theme=gr.themes.Soft(), css=UI_CSS)
        return

    df, msg = run_suite(
        model_ref=args.model_ref,
        config_name=args.config,
        device=args.device,
        dtype=args.dtype,
        results_dir=args.results_dir,
        warmup=args.warmup,
        repeat=args.repeat,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        save_format=args.save_format,
        gpu_verify=args.gpu_verify,
        sage_cap=args.sage_cap,
        sage_runs=args.sage_runs,
        sage_iters=args.sage_iters,
        sage_data_size=args.sage_data_size,
        sage_threshold_sigma_k=args.sage_threshold_sigma_k,
    )
    print(msg)
    print(_df_for_display(df).to_string(index=False))


if __name__ == "__main__":
    main()

