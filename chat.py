import hashlib
import json
import os
import glob
import shutil
import subprocess
import sys
import tempfile
import traceback
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import gc

import gradio as gr

from cove_paths import (
    DEFAULT_LLAMA_MODEL_REF,
    REPO_ROOT,
    canonical_path,
    dp_low_freq_words_rel,
    dp_nearest_tokens_npz_rel,
    env_str,
    migrate_legacy_repo_paths_in_tree,
    paths_equivalent,
    repo_path,
    rewrite_legacy_paths_in_text,
    sage_enroll_iters,
    sage_enroll_runs,
    sage_profile_path,
    sage_verify_threads,
    zkllm_chat_dir,
)
from cove_ui_theme import build_gradio_css

migrate_legacy_repo_paths_in_tree(os.path.join(REPO_ROOT, "zk-PIM"))
ZKLLM_DIR = os.path.join(REPO_ROOT, "zk-PIM")
SAGE_DIR = os.path.join(REPO_ROOT, "LSAGE")
SAGE_PROFILE = sage_profile_path()
FORCE_MATH_SDP_ENV = "COVE_FORCE_MATH_SDP"
UI_CSS = build_gradio_css(max_width="1400px", with_chat_extras=True)

_TORCH = None


def _torch():
    """Lazy import so SAGE gpu_attest can run before PyTorch grabs the GPU."""
    global _TORCH
    if _TORCH is None:
        import torch as _t

        _TORCH = _t
    return _TORCH


def _force_math_sdp() -> None:
    try:
        torch = _torch()
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


def _maybe_force_math_sdp_for_chat() -> None:
    if _math_sdp_env_enabled():
        _force_math_sdp()
        return
    try:
        torch = _torch()
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(torch.device(STATE.device).index or 0)
            if "H20" in str(name).upper():
                _force_math_sdp()
    except Exception:
        pass


@dataclass
class AppState:
    tokenizer: Optional[object] = None
    model: Optional[object] = None
    model_ref: Optional[str] = None
    device: str = "cuda:0"
    low_freq_ids: Optional[set] = None
    nearest_neighbors: Optional[object] = None
    nearest_scores: Optional[object] = None
    cuda_touched: bool = False


STATE = AppState()


def current_python() -> str:
    return sys.executable or "python"


def zkllm_subprocess_python() -> str:
    """
    Interpreter for zkLLM subprocesses (ppgen/commit/prompt_to_layer_input/llama-*).
    Set COVE_ZKLLM_PYTHON to override; otherwise use a known Anaconda build when present
    so behavior matches manual runs (e.g. /home/baijiaoyang/anaconda3/bin/python ...).
    """
    override = env_str("COVE_ZKLLM_PYTHON", "CVEE_ZKLLM_PYTHON")
    if override:
        return override
    candidate = "/home/baijiaoyang/anaconda3/bin/python"
    try:
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    except OSError:
        pass
    return current_python()


def _subproc_text(chunk: Optional[object]) -> str:
    """Normalize subprocess streams to str (TimeoutExpired may attach bytes even when text=True)."""
    if chunk is None:
        return ""
    if isinstance(chunk, bytes):
        return chunk.decode("utf-8", errors="replace")
    return str(chunk)


def _zkllm_prep_timeout_sec() -> Optional[float]:
    """llama-ppgen / llama-commit are long-running. Default 4h; set COVE_ZKLLM_PREP_TIMEOUT_SEC=0 for no limit."""
    raw = env_str("COVE_ZKLLM_PREP_TIMEOUT_SEC", "CVEE_ZKLLM_PREP_TIMEOUT_SEC")
    if raw == "0":
        return None
    if raw == "":
        return float(4 * 3600)
    return float(raw)


def _zkllm_commit_timeout_sec() -> Optional[float]:
    """
    llama-commit.py alone can run much longer than ppgen (often many hours on one GPU).
    COVE_ZKLLM_COMMIT_TIMEOUT_SEC: seconds, or 0 = no limit.
    If unset: default 24h (86400). Set COVE_ZKLLM_PREP_TIMEOUT_SEC only for step 1; commit uses this.
    """
    raw = env_str("COVE_ZKLLM_COMMIT_TIMEOUT_SEC", "CVEE_ZKLLM_COMMIT_TIMEOUT_SEC")
    if raw == "0":
        return None
    if raw == "":
        return float(24 * 3600)
    return float(raw)


def run_cmd(cmd: list[str], cwd: Optional[str] = None, timeout_sec: Optional[float] = 180, env: Optional[dict] = None) -> tuple[int, str]:
    try:
        p = subprocess.run(
            cmd,
            cwd=cwd,
            text=True,
            capture_output=True,
            timeout=timeout_sec,
            check=False,
            env=env,
        )
        out = (p.stdout or "") + ("\n" + p.stderr if p.stderr else "")
        return p.returncode, out
    except FileNotFoundError as e:
        return 127, f"Command not found: {e.filename}"
    except subprocess.TimeoutExpired as e:
        tail = _subproc_text(e.stdout)
        if e.stderr is not None:
            tail += "\n" + _subproc_text(e.stderr)
        lim = "no limit" if timeout_sec is None else f"{timeout_sec}s"
        return 124, f"Timeout after {lim}\n{tail}"


def ensure_model_loaded(model_size: int, model_path: str, cache_dir: str) -> str:
    from zkllm.model_load_utils import load_tokenizer_and_model, resolve_model_ref

    torch = _torch()
    _maybe_force_math_sdp_for_chat()
    model_ref = resolve_model_ref(model_size, repo_path(model_path))
    if STATE.model is not None and STATE.tokenizer is not None and STATE.model_ref == model_ref:
        return "Model cache hit."

    tokenizer, model = load_tokenizer_and_model(model_ref, cache_dir=repo_path(cache_dir), local_files_only=True)
    model.eval()
    model.to(torch.device(STATE.device))
    STATE.tokenizer = tokenizer
    STATE.model = model
    STATE.model_ref = model_ref
    STATE.cuda_touched = True
    return f"Loaded model from {model_ref}."


def maybe_apply_dp(input_ids_tensor, dp_enable: bool, dp_epsilon: float, dp_sensitivity: float, dp_noise_type: str,
                   dp_replace_prob: float, dp_gauss_noise_factor: float, low_freq_words_txt: str, nearest_tokens_npz: str):
    from zkllm.prompt_to_layer_input import (
        _load_low_freq_ids,
        _load_nearest_index_npz,
        privatize_input_ids,
    )

    torch = _torch()
    if not dp_enable:
        return input_ids_tensor, "DP disabled."

    if STATE.low_freq_ids is None:
        STATE.low_freq_ids = _load_low_freq_ids(low_freq_words_txt)
    if STATE.nearest_neighbors is None or STATE.nearest_scores is None:
        nn, sc = _load_nearest_index_npz(nearest_tokens_npz)
        STATE.nearest_neighbors = nn
        STATE.nearest_scores = sc

    ids_list = input_ids_tensor[0].tolist()
    private_ids = privatize_input_ids(
        ids_list,
        low_freq_ids=STATE.low_freq_ids,
        nearest_neighbors=STATE.nearest_neighbors,
        nearest_scores=STATE.nearest_scores,
        epsilon=dp_epsilon,
        sensitivity=dp_sensitivity,
        noise_type=dp_noise_type,
        replace_prob_non_low_freq=dp_replace_prob,
        gauss_noise_factor=dp_gauss_noise_factor,
    )
    out = torch.tensor([private_ids], device=input_ids_tensor.device, dtype=input_ids_tensor.dtype)
    changed = sum(1 for a, b in zip(ids_list, private_ids) if a != b)
    return out, f"DP enabled. replaced_tokens={changed}/{len(ids_list)}"


def zkllm_workdir(model_size: int) -> str:
    return os.path.join(ZKLLM_DIR, "zkllm-workdir", f"Llama-2-{model_size}b")


def _prompt_slug(prompt: str, max_len: int = 48) -> str:
    """Filesystem-safe short label derived from the user prompt."""
    text = (prompt or "").strip()
    if not text:
        return "empty"
    slug = re.sub(r"\s+", "_", text)
    slug = re.sub(r'[\\/:*?"<>|\x00-\x1f]+', "", slug)
    slug = slug.strip("._")
    if not slug:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
        return f"p_{digest}"
    if len(slug) > max_len:
        slug = slug[:max_len].rstrip("._")
    return slug or "prompt"


def _zkllm_run_workdir(prompt: str) -> str:
    """Named run folder under zkllm-chat/: <time>_<prompt-slug>."""
    root = zkllm_chat_dir()
    os.makedirs(root, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = _prompt_slug(prompt)
    base = f"{stamp}_{slug}"
    path = os.path.join(root, base)
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=False)
        return path
    for i in range(2, 1000):
        path = os.path.join(root, f"{base}_{i}")
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=False)
            return path
    return tempfile.mkdtemp(prefix=f"{base}_", dir=root)


def _write_zkllm_run_metadata(
    workdir: str,
    *,
    prompt: str,
    model_size: int,
    model_path: str,
    cache_dir: str,
    seq_len: int,
    dp_enable: bool,
) -> None:
    with open(os.path.join(workdir, "prompt.txt"), "w", encoding="utf-8") as f:
        f.write(prompt)
    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "prompt": prompt,
        "prompt_slug": _prompt_slug(prompt),
        "model_size": model_size,
        "model_path": repo_path(model_path),
        "cache_dir": repo_path(cache_dir),
        "seq_len": seq_len,
        "dp_enable": dp_enable,
    }
    with open(os.path.join(workdir, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def resolve_nvcc() -> Optional[str]:
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


def _subprocess_env_base() -> dict:
    """NVCC/PATH/conda for child processes without touching PyTorch CUDA."""
    env = os.environ.copy()
    nvcc = resolve_nvcc()
    if nvcc:
        env["NVCC"] = nvcc
        env["PATH"] = f"{os.path.dirname(nvcc)}:{env.get('PATH', '')}"

    conda_prefix = env.get("CONDA_PREFIX", "")
    if conda_prefix and not os.path.isdir(conda_prefix):
        env["CONDA_PREFIX"] = sys.prefix
    elif conda_prefix and not os.path.exists(os.path.join(conda_prefix, "bin", "nvcc")) and sys.prefix:
        env["CONDA_PREFIX"] = sys.prefix
    return env


def _sage_subprocess_env() -> dict:
    """SAGE gpu_attest env — must not call torch.cuda (would grab GPU before attest)."""
    env = _subprocess_env_base()
    env.setdefault("SM", "90")
    return env


def zkllm_subprocess_env() -> dict:
    env = _subprocess_env_base()
    try:
        torch = _torch()
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(torch.cuda.current_device())
            sm = major * 10 + minor
            env.setdefault("SM", "90" if sm >= 120 else str(sm))
    except Exception:
        pass
    return env


def _zkllm_artifact_manifest_path(model_size: int) -> str:
    return os.path.join(zkllm_workdir(model_size), ".artifact_manifest.json")


def _zkllm_read_manifest(model_size: int) -> Optional[dict]:
    man = _zkllm_artifact_manifest_path(model_size)
    if not os.path.isfile(man):
        return None
    try:
        with open(man, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None


def _zkllm_same_weights(model_size: int, model_path: str) -> bool:
    d = _zkllm_read_manifest(model_size)
    if not d:
        return False
    return (
        int(d.get("model_size", -1)) == int(model_size)
        and canonical_path(str(d.get("model_path", ""))) == canonical_path(model_path)
    )


def _zkllm_manifest_matches(model_size: int, model_path: str, cache_dir: str) -> bool:
    """Commitments are tied to model_path; cache_dir may change after repo rename."""
    mp = canonical_path(model_path)
    cd = canonical_path(cache_dir)
    d = _zkllm_read_manifest(model_size)
    if not d:
        return False
    if int(d.get("model_size", -1)) != int(model_size):
        return False
    if canonical_path(str(d.get("model_path", ""))) != mp:
        return False
    stored_cd = str(d.get("cache_dir", ""))
    if paths_equivalent(stored_cd, cd):
        return True
    stored_canon = canonical_path(stored_cd)
    if stored_canon and not os.path.exists(stored_canon):
        return True
    return False


def _write_zkllm_artifact_manifest(model_size: int, model_path: str, cache_dir: str) -> None:
    man = _zkllm_artifact_manifest_path(model_size)
    os.makedirs(os.path.dirname(man), exist_ok=True)
    payload = {
        "model_size": int(model_size),
        "model_path": canonical_path(model_path),
        "cache_dir": canonical_path(cache_dir),
    }
    with open(man, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def ensure_zkllm_artifacts(model_size: int, model_path: str, cache_dir: str) -> tuple[bool, str]:
    """
    Prepare pp/int/commitment files required by the layer verification binaries.
    The CUDA proof executables expect these files under zk-PIM/zkllm-workdir.
    """
    workdir = zkllm_workdir(model_size)
    pp_files = glob.glob(os.path.join(workdir, "*-pp.bin"))
    commitment_files = glob.glob(os.path.join(workdir, "layer-0-*-commitment.bin"))
    int_files = glob.glob(os.path.join(workdir, "layer-0-*-int.bin"))
    files_ok = bool(pp_files and commitment_files and int_files)
    if files_ok and _zkllm_manifest_matches(model_size, model_path, cache_dir):
        msg = f"zkLLM artifacts cache hit: {workdir}"
        d = _zkllm_read_manifest(model_size) or {}
        stored_cd = canonical_path(str(d.get("cache_dir", "")))
        want_cd = canonical_path(cache_dir)
        if stored_cd != want_cd:
            _write_zkllm_artifact_manifest(model_size, model_path, cache_dir)
            msg += f"\n(updated manifest cache_dir: {stored_cd} → {want_cd})"
        return True, msg

    logs = []
    if files_ok and not _zkllm_manifest_matches(model_size, model_path, cache_dir):
        if _zkllm_same_weights(model_size, model_path):
            _write_zkllm_artifact_manifest(model_size, model_path, cache_dir)
            return True, (
                f"zkLLM artifacts cache hit (manifest updated, same weights; no pp/commit regen): {workdir}"
            )
        busy = _sage_gpu_busy_detail()
        if busy:
            return False, (
                "Refusing to regenerate zkLLM pp/commitments while GPU is busy "
                "(SAGE needs idle GPU for ~0.35s runtime).\n" + busy
            )
        logs.append(
            "zkLLM artifacts exist but do not match this model_path "
            f"(see {_zkllm_artifact_manifest_path(model_size)}). Regenerating pp/commitments."
        )

    logs.append(f"Preparing zkLLM artifacts under: {workdir}")
    env = zkllm_subprocess_env()
    common = ["--model_path", repo_path(model_path), "--cache_dir", repo_path(cache_dir)]
    prep_steps = [
        [zkllm_subprocess_python(), os.path.join(ZKLLM_DIR, "llama-ppgen.py"), str(model_size), *common],
        [zkllm_subprocess_python(), os.path.join(ZKLLM_DIR, "llama-commit.py"), str(model_size), "16", *common],
    ]
    timeouts = [_zkllm_prep_timeout_sec(), _zkllm_commit_timeout_sec()]
    for i, cmd in enumerate(prep_steps, start=1):
        code, out = run_cmd(cmd, cwd=ZKLLM_DIR, timeout_sec=timeouts[i - 1], env=env)
        logs.append(f"[prepare {i}] {' '.join(cmd)}\nexit={code}\n{out[-2000:]}")
        if code != 0:
            if code == 124:
                logs.append(
                    "\nHint: exit 124 means subprocess TIMEOUT. llama-commit.py often needs many hours. "
                    "Default commit timeout is 24h (COVE_ZKLLM_COMMIT_TIMEOUT_SEC unset). "
                    "Set COVE_ZKLLM_COMMIT_TIMEOUT_SEC=0 for no limit, or a larger value in seconds; "
                    "or run llama-commit.py in a terminal/tmux and retry zkLLM when finished.\n"
                )
            return False, "\n".join(logs)

    pp_files = glob.glob(os.path.join(workdir, "*-pp.bin"))
    commitment_files = glob.glob(os.path.join(workdir, "layer-0-*-commitment.bin"))
    int_files = glob.glob(os.path.join(workdir, "layer-0-*-int.bin"))
    if not (pp_files and commitment_files and int_files):
        logs.append("Artifact preparation finished, but required layer-0 files are still missing.")
        return False, "\n".join(logs)
    _write_zkllm_artifact_manifest(model_size, model_path, cache_dir)
    logs.append(f"Prepared zkLLM artifacts: pp={len(pp_files)}, commitments={len(commitment_files)}, int={len(int_files)}")
    return True, "\n".join(logs)


def generate_layer_input_bin(
    prompt: str,
    seq_len: int,
    output_file: str,
    *,
    model_size: int,
    model_path: str,
    cache_dir: str,
    dp_enable: bool,
    dp_epsilon: float,
    dp_sensitivity: float,
    dp_noise_type: str,
    dp_replace_prob: float,
    dp_gauss_noise_factor: float,
    low_freq_words_txt: str,
    nearest_tokens_npz: str,
) -> tuple[bool, str]:
    """
    Keep behavior aligned with user's proven manual workflow:
    call prompt_to_layer_input.py directly to generate layer_input.bin.
    """
    cmd = [
        zkllm_subprocess_python(),
        os.path.join(ZKLLM_DIR, "prompt_to_layer_input.py"),
        str(model_size),
        "--model_path",
        repo_path(model_path),
        "--cache_dir",
        repo_path(cache_dir),
        "--prompt",
        prompt,
        "--seq_len",
        str(seq_len),
        "--output_file",
        output_file,
        "--log_sf",
        "16",
    ]
    if dp_enable:
        cmd.extend(
            [
                "--dp_enable",
                "--dp_epsilon",
                str(dp_epsilon),
                "--dp_sensitivity",
                str(dp_sensitivity),
                "--dp_noise_type",
                dp_noise_type,
                "--dp_replace_prob",
                str(dp_replace_prob),
                "--dp_gauss_noise_factor",
                str(dp_gauss_noise_factor),
                "--low_freq_words_txt",
                repo_path(low_freq_words_txt),
                "--nearest_tokens_npz",
                repo_path(nearest_tokens_npz),
            ]
        )

    code, out = run_cmd(cmd, cwd=ZKLLM_DIR, timeout_sec=1800, env=zkllm_subprocess_env())
    if code == 0 and os.path.isfile(output_file):
        return True, out[-1200:]
    return False, out[-3000:]


def _sage_gpu_busy_detail() -> Optional[str]:
    """
    Return a human-readable reason if the GPU is too busy for timing attestation (~0.35s on idle H20).
    zkLLM commit-param / llama-commit after a repo rename often causes ~0.76s+ runtime.
    """
    blockers: list[str] = []
    code, apps = run_cmd(
        ["nvidia-smi", "--query-compute-apps=process_name,used_gpu_memory", "--format=csv,noheader"],
        timeout_sec=15,
    )
    if code == 0 and apps.strip():
        for line in apps.splitlines():
            name = line.split(",")[0].strip().lower()
            if any(k in name for k in ("commit-param", "ppgen", "llama-commit", "llama-ppgen")):
                blockers.append(line.strip())
    code, util = run_cmd(
        ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
        timeout_sec=15,
    )
    gpu_util = -1
    if code == 0 and util.strip():
        try:
            gpu_util = int(float(util.strip().split("\n")[0]))
        except (TypeError, ValueError):
            pass
    if blockers:
        return (
            "GPU is running zkLLM weight preparation (commit-param / llama-commit). "
            "This often starts after renaming the repo when artifact manifest no longer matches.\n"
            "Processes:\n  " + "\n  ".join(blockers[:6]) + "\n"
            "Stop them (or wait until commit finishes), confirm `nvidia-smi` is idle, then retry SAGE."
        )
    if gpu_util >= 50:
        return (
            f"GPU utilization is {gpu_util}% (SAGE timing needs an idle GPU for ~0.35s runtime). "
            "Check `nvidia-smi`, stop unrelated jobs, then retry."
        )
    return None


def _sage_profile_header() -> str:
    """Show which profile file supplies baseline_mean (not the single-shot runtime= line)."""
    path = sage_profile_path()
    if not os.path.isfile(path):
        return f"[SAGE profile] missing: {path}\n(run enroll once to set baseline_mean)\n"
    try:
        with open(path, encoding="utf-8") as f:
            p = json.load(f)
        return (
            f"[SAGE profile] {path}\n"
            f"baseline_mean={float(p.get('baseline_mean', 0)):.6f}s "
            f"runtime_upper={float(p.get('runtime_upper', 0)):.6f}s "
            f"iters={p.get('iters')} verify_threads={p.get('verify_threads')}\n"
        )
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return f"[SAGE profile] {path} (unreadable)\n"


def run_sage_gpu_check() -> str:
    # Use LSAGE/run_attestation.py as Cove's GPU verification backend.
    if not os.path.isdir(SAGE_DIR):
        return "SAGE status: SKIPPED (LSAGE not found)"
    if resolve_nvcc() is None:
        return run_gpu_baseline_check(
            reason="SAGE attestation requires nvcc, but nvcc was not found",
            detail="Expected CUDA compiler at /usr/local/cuda/bin/nvcc or in PATH.",
        )

    busy = _sage_gpu_busy_detail()
    if busy:
        return f"SAGE status: FAIL\n[SAGE profile] {SAGE_PROFILE}\n{busy}"

    # Never use zkllm_subprocess_env() here — it initializes torch.cuda in this process.
    env = _sage_subprocess_env()
    profile_arg = ["--profile", SAGE_PROFILE]
    prefix = _sage_profile_header()
    if not os.path.isfile(SAGE_PROFILE):
        enroll_cmd = [
            current_python(),
            os.path.join(SAGE_DIR, "run_attestation.py"),
            "enroll",
            *profile_arg,
            "--gpu-model",
            "H20",
            "--cap",
            "90",
            "--runs",
            str(sage_enroll_runs()),
            "--iters",
            str(sage_enroll_iters()),
            "--data-size",
            str(1 << 20),
            "--threshold-sigma-k",
            "2.0",
            "--verify-threads",
            str(sage_verify_threads()),
        ]
        code, out = run_cmd(enroll_cmd, cwd=SAGE_DIR, timeout_sec=1800, env=env)
        prefix += "[auto-enroll]\n" + rewrite_legacy_paths_in_text(out[-2000:]) + "\n\n"
        if code != 0:
            return f"SAGE status: ENROLL_FAILED\n{prefix}"
        prefix = _sage_profile_header() + prefix

    sage_vt = sage_verify_threads()
    attest_cmd = [
        current_python(),
        os.path.join(SAGE_DIR, "run_attestation.py"),
        "attest",
        *profile_arg,
        "--verify-threads",
        str(sage_vt),
    ]
    code, out = run_cmd(attest_cmd, cwd=SAGE_DIR, timeout_sec=1800, env=env)
    status = "PASS" if code == 0 and "verification SUCCEED" in out else "FAIL"
    body = rewrite_legacy_paths_in_text(f"{out[-3000:]}")
    if status == "FAIL" and "time_ok=False" in out:
        extra = _sage_gpu_busy_detail() or (
            "Runtime exceeded profile runtime_upper while checksum may still pass. "
            "On an idle H20 expect runtime≈0.35s; ~0.76s usually means another GPU job is active."
        )
        body = extra + "\n\n" + body
    return f"SAGE status: {status}\n{prefix}{body}"


def run_gpu_baseline_check(reason: str, detail: str) -> str:
    """
    Fallback integrity signal when strict SAGE checksum is unavailable.
    This is NOT equivalent to SAGE strict attestation, but still provides
    a practical runtime health check for engineering deployment.
    """
    dev_info = "unknown"
    torch_check = "unknown"
    try:
        torch = _torch()
        if torch.cuda.is_available():
            idx = torch.cuda.current_device()
            name = torch.cuda.get_device_name(idx)
            cap = torch.cuda.get_device_capability(idx)
            # lightweight CUDA op sanity test
            x = torch.randn(1024, device=f"cuda:{idx}")
            y = (x * 2.0).sum().item()
            dev_info = f"device={name}, capability={cap}"
            torch_check = f"torch_cuda_op=ok, sample_sum={y:.4f}"
        else:
            dev_info = "cuda_not_available"
            torch_check = "torch_cuda_op=skipped"
    except Exception as e:
        torch_check = f"torch_cuda_op=error: {e}"

    code, smi = run_cmd(["nvidia-smi"], timeout_sec=30)
    smi_tail = smi[-1200:] if smi else "(nvidia-smi output empty)"
    smi_state = "ok" if code == 0 else f"error(code={code})"

    return (
        "SAGE status: DEGRADED\n"
        f"reason: {reason}\n"
        f"{dev_info}\n"
        f"{torch_check}\n"
        f"nvidia_smi={smi_state}\n"
        "strict_sage_detail_tail:\n"
        f"{detail[-1200:]}\n"
        "nvidia_smi_tail:\n"
        f"{smi_tail}"
    )


_ZKLLM_PASS_MARKERS = (
    "Self attention proof successfully verified!",
    "SwiGLU proof complete.",
    "QKV linear proof successfully verified!",
)


def _zkllm_format_status_log(logs: list[str], *, max_chars: int = 3500) -> str:
    """User-facing log: drop noisy tqdm bars when possible; rewrite Cvee→Cove paths."""
    merged = rewrite_legacy_paths_in_text("\n".join(logs))
    keys = (
        "zkLLM status",
        "Run workdir",
        "cache hit",
        "artifact",
        "[prepare",
        "[step ",
        "exit=",
        "proof",
        "verified",
        "complete",
        "Opening",
        "Hint:",
    )
    lines = [ln for ln in merged.splitlines() if any(k in ln for k in keys)]
    if len(lines) >= 8:
        body = "\n".join(lines)
        return body[-max_chars:] if len(body) > max_chars else body
    return merged[-max_chars:] if len(merged) > max_chars else merged


def _zkllm_pipeline_passed(logs: list[str], step_codes: list[int]) -> bool:
    merged = "\n".join(logs)
    if not all(c == 0 for c in step_codes):
        return False
    return any(m in merged for m in _ZKLLM_PASS_MARKERS)


def run_zkllm_layer_check(
    model_size: int,
    model_path: str,
    seq_len: int,
    prompt_for_input: str,
    *,
    cache_dir: str,
    dp_enable: bool,
    dp_epsilon: float,
    dp_sensitivity: float,
    dp_noise_type: str,
    dp_replace_prob: float,
    dp_gauss_noise_factor: float,
    low_freq_words_txt: str,
    nearest_tokens_npz: str,
) -> str:
    # Run a short layer-0 pipeline with generated files to show zkLLM verification status.
    work = _zkllm_run_workdir(prompt_for_input)
    _write_zkllm_run_metadata(
        work,
        prompt=prompt_for_input,
        model_size=model_size,
        model_path=model_path,
        cache_dir=cache_dir,
        seq_len=seq_len,
        dp_enable=dp_enable,
    )
    input_file = os.path.join(work, "layer_input.bin")
    attn_input = os.path.join(work, "attn_input.bin")
    attn_output = os.path.join(work, "attn_output.bin")
    post_attn_norm_input = os.path.join(work, "post_attn_norm_input.bin")
    ffn_input = os.path.join(work, "ffn_input.bin")
    ffn_output = os.path.join(work, "ffn_output.bin")
    output_file = os.path.join(work, "layer_output.bin")

    artifacts_ok, artifacts_msg = ensure_zkllm_artifacts(
        model_size=model_size,
        model_path=model_path,
        cache_dir=cache_dir,
    )
    if not artifacts_ok:
        return f"zkLLM status: PREP_FAILED\n{artifacts_msg}"

    # Use the actual chat prompt as zkLLM layer input source.
    # This makes zk verification input consistent with the UI request.
    ok, msg = generate_layer_input_bin(
        prompt_for_input,
        seq_len,
        input_file,
        model_size=model_size,
        model_path=model_path,
        cache_dir=cache_dir,
        dp_enable=dp_enable,
        dp_epsilon=dp_epsilon,
        dp_sensitivity=dp_sensitivity,
        dp_noise_type=dp_noise_type,
        dp_replace_prob=dp_replace_prob,
        dp_gauss_noise_factor=dp_gauss_noise_factor,
        low_freq_words_txt=low_freq_words_txt,
        nearest_tokens_npz=nearest_tokens_npz,
    )
    if not ok:
        return f"zkLLM status: PREP_FAILED\n{artifacts_msg}\n{msg}"

    env = zkllm_subprocess_env()
    py = zkllm_subprocess_python()
    steps = [
        [py, os.path.join(ZKLLM_DIR, "llama-rmsnorm.py"), str(model_size), "0", "input", str(seq_len), "--input_file", input_file, "--output_file", attn_input, "--model_path", repo_path(model_path), "--cache_dir", repo_path(cache_dir)],
        [py, os.path.join(ZKLLM_DIR, "llama-self-attn.py"), str(model_size), "0", str(seq_len), "--input_file", attn_input, "--output_file", attn_output, "--model_path", repo_path(model_path), "--cache_dir", repo_path(cache_dir)],
        [py, os.path.join(ZKLLM_DIR, "llama-skip-connection.py"), "--block_input_file", input_file, "--block_output_file", attn_output, "--output_file", post_attn_norm_input],
        [py, os.path.join(ZKLLM_DIR, "llama-rmsnorm.py"), str(model_size), "0", "post_attention", str(seq_len), "--input_file", post_attn_norm_input, "--output_file", ffn_input, "--model_path", repo_path(model_path), "--cache_dir", repo_path(cache_dir)],
        [py, os.path.join(ZKLLM_DIR, "llama-ffn.py"), str(model_size), "0", str(seq_len), "--input_file", ffn_input, "--output_file", ffn_output, "--model_path", repo_path(model_path), "--cache_dir", repo_path(cache_dir)],
        [py, os.path.join(ZKLLM_DIR, "llama-skip-connection.py"), "--block_input_file", post_attn_norm_input, "--block_output_file", ffn_output, "--output_file", output_file],
    ]

    logs = [f"Run workdir: {work}", artifacts_msg, msg]
    step_codes: list[int] = []
    for i, cmd in enumerate(steps, start=1):
        code, out = run_cmd(cmd, cwd=ZKLLM_DIR, timeout_sec=1200, env=env)
        step_codes.append(code)
        cmd_display = rewrite_legacy_paths_in_text(" ".join(cmd))
        logs.append(f"[step {i}] {cmd_display}\nexit={code}\n{out[-1200:]}\n")
        if code != 0:
            if "D or N is not power of 2, or D is not divisible by N" in out:
                logs.append(
                    "\nHint: current seq_len likely violates tLookup constraints in FFN proof.\n"
                    "Use README-aligned setting: sequence_length=2048 (power of two), and regenerate input with the same seq_len.\n"
                )
            return "zkLLM status: FAIL\n" + _zkllm_format_status_log(logs)

    if _zkllm_pipeline_passed(logs, step_codes):
        return "zkLLM status: PASS\n" + _zkllm_format_status_log(logs)
    return "zkLLM status: UNKNOWN (pipeline ran but no explicit marker)\n" + _zkllm_format_status_log(logs)


def status_to_badge(status_text: str, *, kind: str) -> str:
    """
    Convert status strings returned by run_sage_gpu_check / run_zkllm_layer_check into colored badges.
    kind: 'gpu' or 'zkllm'
    """
    txt = (status_text or "").strip()
    # GPU
    if kind == "gpu":
        if "SAGE status: PASS" in txt:
            return '<div style="padding:4px 10px;border-radius:999px;background:#1a8f4a;color:white;font-weight:700;display:inline-block;">GPU PASS</div>'
        if "SAGE status: DEGRADED" in txt:
            return '<div style="padding:4px 10px;border-radius:999px;background:#d48b00;color:white;font-weight:700;display:inline-block;">GPU DEGRADED</div>'
        if "SAGE status: FAIL" in txt:
            return '<div style="padding:4px 10px;border-radius:999px;background:#b42318;color:white;font-weight:700;display:inline-block;">GPU FAIL</div>'
        return '<div style="padding:4px 10px;border-radius:999px;background:#6b7280;color:white;font-weight:700;display:inline-block;">GPU UNKNOWN</div>'
    # zkLLM
    if "zkLLM status: PASS" in txt:
        return '<div style="padding:4px 10px;border-radius:999px;background:#1a8f4a;color:white;font-weight:700;display:inline-block;">zkLLM PASS</div>'
    if "zkLLM status: FAIL" in txt:
        return '<div style="padding:4px 10px;border-radius:999px;background:#b42318;color:white;font-weight:700;display:inline-block;">zkLLM FAIL</div>'
    if "zkLLM status: UNKNOWN" in txt:
        return '<div style="padding:4px 10px;border-radius:999px;background:#d48b00;color:white;font-weight:700;display:inline-block;">zkLLM UNKNOWN</div>'
    if "zkLLM status: PREP_FAILED" in txt:
        return '<div style="padding:4px 10px;border-radius:999px;background:#b42318;color:white;font-weight:700;display:inline-block;">zkLLM PREP_FAILED</div>'
    if "zkLLM status: ERROR" in txt:
        return '<div style="padding:4px 10px;border-radius:999px;background:#b42318;color:white;font-weight:700;display:inline-block;">zkLLM ERROR</div>'
    return '<div style="padding:4px 10px;border-radius:999px;background:#6b7280;color:white;font-weight:700;display:inline-block;">zkLLM UNKNOWN</div>'


def release_model_cache(*, touch_cuda: bool = True) -> str:
    """
    Release loaded model/tokenizer. Only touch CUDA when touch_cuda=True (frees VRAM but
    keeps a PyTorch context on the GPU, which slows the next SAGE gpu_attest).
    """
    had_model = STATE.model is not None
    STATE.model = None
    STATE.tokenizer = None
    STATE.model_ref = None
    if touch_cuda and had_model:
        try:
            torch = _torch()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            gc.collect()
            STATE.cuda_touched = True
        except Exception:
            pass
    return "Released in-process model cache." + (" (CUDA cache cleared)" if touch_cuda and had_model else "")


def run_chat_only(
    user_prompt: str,
    model_path: str,
    model_size: int,
    cache_dir: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    dp_enable: bool,
    dp_epsilon: float,
    dp_sensitivity: float,
    dp_noise_type: str,
    dp_replace_prob: float,
    dp_gauss_noise_factor: float,
    low_freq_words_txt: str,
    nearest_tokens_npz: str,
    run_gpu_check: bool,
):
    try:
        if not user_prompt or not user_prompt.strip():
            user_prompt = "Hello"

        # Run SAGE before inference; release any prior chat model so gpu_attest sees an idle GPU.
        gpu_status = "GPU check skipped."
        release_msg = ""
        if run_gpu_check:
            had_model = STATE.model is not None
            release_msg = release_model_cache(touch_cuda=had_model)
            gpu_status = run_sage_gpu_check()

        load_msg = ensure_model_loaded(model_size, model_path, cache_dir)
        tokenizer = STATE.tokenizer
        model = STATE.model
        assert tokenizer is not None and model is not None

        torch = _torch()
        enc = tokenizer(user_prompt, return_tensors="pt", add_special_tokens=True)
        input_ids = enc["input_ids"].to(torch.device(STATE.device))
        input_ids, dp_msg = maybe_apply_dp(
            input_ids,
            dp_enable=dp_enable,
            dp_epsilon=dp_epsilon,
            dp_sensitivity=dp_sensitivity,
            dp_noise_type=dp_noise_type,
            dp_replace_prob=dp_replace_prob,
            dp_gauss_noise_factor=dp_gauss_noise_factor,
            low_freq_words_txt=low_freq_words_txt,
            nearest_tokens_npz=nearest_tokens_npz,
        )

        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True if temperature > 0 else False,
                temperature=max(temperature, 1e-5),
                top_p=top_p,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        # Decode only newly generated tokens to avoid prompt echo/noisy token artifacts.
        prompt_len = input_ids.shape[1]
        gen_ids = out[0][prompt_len:]
        if gen_ids.numel() == 0:
            decoded = "(no new tokens generated)"
        else:
            decoded = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        meta = f"{release_msg}\n{load_msg}\n{dp_msg}".strip()
        gpu_badge = status_to_badge(gpu_status, kind="gpu")
        return decoded, gpu_badge, gpu_status, meta
    except Exception:
        err = traceback.format_exc()
        gpu_status = f"GPU status: ERROR\n{err}"
        gpu_badge = status_to_badge(gpu_status, kind="gpu")
        return "", gpu_badge, gpu_status, "Failed."


def run_zkllm_only(
    prompt_for_input: str,
    model_path: str,
    model_size: int,
    cache_dir: str,
    zkllm_seq_len: int,
    dp_enable: bool,
    dp_epsilon: float,
    dp_sensitivity: float,
    dp_noise_type: str,
    dp_replace_prob: float,
    dp_gauss_noise_factor: float,
    low_freq_words_txt: str,
    nearest_tokens_npz: str,
    run_gpu_check: bool,
):
    try:
        if not prompt_for_input or not prompt_for_input.strip():
            prompt_for_input = "Hello"

        had_model = STATE.model is not None
        release_msg = release_model_cache(touch_cuda=had_model)

        gpu_status = "GPU check skipped."
        if run_gpu_check:
            gpu_status = run_sage_gpu_check()

        zk_status = run_zkllm_layer_check(
            model_size=model_size,
            model_path=model_path,
            seq_len=zkllm_seq_len,
            prompt_for_input=prompt_for_input,
            cache_dir=cache_dir,
            dp_enable=dp_enable,
            dp_epsilon=dp_epsilon,
            dp_sensitivity=dp_sensitivity,
            dp_noise_type=dp_noise_type,
            dp_replace_prob=dp_replace_prob,
            dp_gauss_noise_factor=dp_gauss_noise_factor,
            low_freq_words_txt=low_freq_words_txt,
            nearest_tokens_npz=nearest_tokens_npz,
        )

        gpu_badge = status_to_badge(gpu_status, kind="gpu")
        zk_badge = status_to_badge(zk_status, kind="zkllm")
        meta = f"{release_msg}\nzkllm_seq_len={zkllm_seq_len}"
        return gpu_badge, gpu_status, zk_badge, zk_status, meta
    except Exception:
        err = traceback.format_exc()
        gpu_status = f"GPU status: ERROR\n{err}"
        zk_status = f"zkLLM status: ERROR\n{err}"
        gpu_badge = status_to_badge(gpu_status, kind="gpu")
        zk_badge = status_to_badge(zk_status, kind="zkllm")
        return gpu_badge, gpu_status, zk_badge, zk_status, "Failed."


def build_ui():
    with gr.Blocks(title="Cove") as demo:
        gr.HTML(
            """
            <div class="hero">
              <h1>Cove</h1>
              <p>DP-aware chat inference and zkLLM Layer0 verification. Use one workflow at a time.</p>
            </div>
            """
        )

        with gr.Tab("Chat Inference (DP + Chat)"):
            with gr.Group(elem_classes=["section-card"]):
                gr.Markdown("### Chat Configuration")
                chat_prompt = gr.Textbox(label="Prompt", lines=5, value="Briefly explain differential privacy.", elem_classes=["full-width"])
                chat_send = gr.Button(">>> START CHAT INFERENCE <<<", variant="primary", elem_id="chat-start-btn", elem_classes=["primary-btn"])
                chat_run_gpu_check = gr.Checkbox(
                    label="Run GPU attestation (SAGE) before inference",
                    value=True,
                )
                chat_answer = gr.Textbox(label="Model response", lines=16, elem_classes=["large-output"])

                with gr.Accordion("Model and DP settings", open=True):
                    chat_model_path = gr.Textbox(label="Model path", value=DEFAULT_LLAMA_MODEL_REF, elem_classes=["full-width"])
                    with gr.Row():
                        chat_model_size = gr.Dropdown(label="Model size", choices=[7, 13], value=7)
                        chat_cache_dir = gr.Textbox(label="cache_dir", value="./model-storage")
                        chat_max_new_tokens = gr.Slider(label="max_new_tokens", minimum=16, maximum=512, value=128, step=8)
                    with gr.Row():
                        chat_temperature = gr.Slider(label="temperature", minimum=0.0, maximum=1.5, value=0.2, step=0.05)
                        chat_top_p = gr.Slider(label="top_p", minimum=0.1, maximum=1.0, value=0.85, step=0.05)
                        chat_dp_enable = gr.Checkbox(label="Enable DP token replacement", value=False)
                    with gr.Row():
                        chat_dp_epsilon = gr.Slider(label="dp_epsilon", minimum=0.1, maximum=200.0, value=100.0, step=0.1)
                        chat_dp_sensitivity = gr.Slider(label="dp_sensitivity", minimum=0.1, maximum=5.0, value=1.0, step=0.1)
                        chat_dp_noise_type = gr.Dropdown(label="dp_noise_type", choices=["laplace", "gaussian"], value="laplace")
                    with gr.Row():
                        chat_dp_replace_prob = gr.Slider(label="dp_replace_prob", minimum=0.0, maximum=1.0, value=0.3, step=0.05)
                        chat_dp_gauss_noise_factor = gr.Slider(label="dp_gauss_noise_factor", minimum=0.1, maximum=10.0, value=2.6, step=0.1)
                    with gr.Row():
                        chat_low_freq_words_txt = gr.Textbox(
                            label="low_freq_words.txt",
                            value=dp_low_freq_words_rel(DEFAULT_LLAMA_MODEL_REF),
                        )
                        chat_nearest_tokens_npz = gr.Textbox(
                            label="nearest_tokens_30.npz",
                            value=dp_nearest_tokens_npz_rel(DEFAULT_LLAMA_MODEL_REF),
                        )

            with gr.Group(elem_classes=["section-card"]):
                gr.Markdown("### Chat Run Status")
                chat_gpu_badge = gr.HTML()
                chat_gpu_status = gr.Textbox(label="GPU attestation result", lines=8, elem_classes=["status-output"])
                chat_meta = gr.Textbox(label="Execution metadata", lines=3, elem_classes=["status-output"])

            chat_send.click(
                fn=run_chat_only,
                inputs=[
                    chat_prompt,
                    chat_model_path,
                    chat_model_size,
                    chat_cache_dir,
                    chat_max_new_tokens,
                    chat_temperature,
                    chat_top_p,
                    chat_dp_enable,
                    chat_dp_epsilon,
                    chat_dp_sensitivity,
                    chat_dp_noise_type,
                    chat_dp_replace_prob,
                    chat_dp_gauss_noise_factor,
                    chat_low_freq_words_txt,
                    chat_nearest_tokens_npz,
                    chat_run_gpu_check,
                ],
                outputs=[chat_answer, chat_gpu_badge, chat_gpu_status, chat_meta],
            )

        with gr.Tab("zkLLM Verification (Layer0)"):
            with gr.Group(elem_classes=["section-card"]):
                gr.Markdown("### Verification Configuration")
                zk_prompt = gr.Textbox(label="Prompt for layer_input.bin", lines=5, value="Briefly explain differential privacy.", elem_classes=["full-width"])
                zk_run = gr.Button(">>> START zkLLM VERIFICATION <<<", variant="primary", elem_id="zkllm-start-btn", elem_classes=["primary-btn"])
                zk_run_gpu_check = gr.Checkbox(
                    label="Run GPU attestation (SAGE) before zkLLM pipeline",
                    value=True,
                )

                with gr.Accordion("Model and DP settings", open=True):
                    zk_model_path = gr.Textbox(label="Model path", value=DEFAULT_LLAMA_MODEL_REF, elem_classes=["full-width"])
                    with gr.Row():
                        zk_model_size = gr.Dropdown(label="Model size", choices=[7, 13], value=7)
                        zk_cache_dir = gr.Textbox(label="cache_dir", value="./model-storage")
                        zk_seq_len = gr.Slider(label="zkLLM seq_len", minimum=64, maximum=4096, value=2048, step=64)
                    with gr.Row():
                        zk_dp_enable = gr.Checkbox(label="Enable DP token replacement for input", value=False)
                        zk_dp_epsilon = gr.Slider(label="dp_epsilon", minimum=0.1, maximum=200.0, value=100.0, step=0.1)
                        zk_dp_sensitivity = gr.Slider(label="dp_sensitivity", minimum=0.1, maximum=5.0, value=1.0, step=0.1)
                    with gr.Row():
                        zk_dp_noise_type = gr.Dropdown(label="dp_noise_type", choices=["laplace", "gaussian"], value="laplace")
                        zk_dp_replace_prob = gr.Slider(label="dp_replace_prob", minimum=0.0, maximum=1.0, value=0.3, step=0.05)
                        zk_dp_gauss_noise_factor = gr.Slider(label="dp_gauss_noise_factor", minimum=0.1, maximum=10.0, value=2.6, step=0.1)
                    with gr.Row():
                        zk_low_freq_words_txt = gr.Textbox(
                            label="low_freq_words.txt",
                            value=dp_low_freq_words_rel(DEFAULT_LLAMA_MODEL_REF),
                        )
                        zk_nearest_tokens_npz = gr.Textbox(
                            label="nearest_tokens_30.npz",
                            value=dp_nearest_tokens_npz_rel(DEFAULT_LLAMA_MODEL_REF),
                        )

            with gr.Group(elem_classes=["section-card"]):
                gr.Markdown("### Verification Status")
                zk_badge = gr.HTML()
                zk_status = gr.Textbox(label="zkLLM verification result", lines=16, elem_classes=["large-output"])
                zk_gpu_badge = gr.HTML()
                zk_gpu_status = gr.Textbox(label="GPU attestation result", lines=8, elem_classes=["status-output"])
                zk_meta = gr.Textbox(label="Execution metadata", lines=3, elem_classes=["status-output"])

            zk_run.click(
                fn=run_zkllm_only,
                inputs=[
                    zk_prompt,
                    zk_model_path,
                    zk_model_size,
                    zk_cache_dir,
                    zk_seq_len,
                    zk_dp_enable,
                    zk_dp_epsilon,
                    zk_dp_sensitivity,
                    zk_dp_noise_type,
                    zk_dp_replace_prob,
                    zk_dp_gauss_noise_factor,
                    zk_low_freq_words_txt,
                    zk_nearest_tokens_npz,
                    zk_run_gpu_check,
                ],
                outputs=[zk_gpu_badge, zk_gpu_status, zk_badge, zk_status, zk_meta],
            )
    return demo


if __name__ == "__main__":
    # Do not touch CUDA here; SAGE gpu_attest needs an idle GPU (see run_chat_only).
    app = build_ui()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False, theme=gr.themes.Soft(), css=UI_CSS)
