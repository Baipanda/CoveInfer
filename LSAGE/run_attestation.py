#!/usr/bin/env python3
import argparse
import json
import math
import os
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BIN = ROOT / "gpu_attest_40xx"
SRC = ROOT / "gpu_attest_40xx.cu"
BUILD_META = ROOT / ".gpu_attest_build.json"
PROFILE_JSON = ROOT / "attestation_profile.json"
RUNTIME_RE = re.compile(r"Runtime:\s*([0-9]+(?:\.[0-9]+)?)\s*s")
GPU_CHK_RE = re.compile(r"checksum on GPU：0x([0-9a-fA-F]+)")
VERIFY_RE = re.compile(r"verification\s+(SUCCEED|FAILED)")


def _raise_if_cuda_program_failed(text: str) -> None:
    """Binary prints CUDA errors to stderr (merged into text). Surface them before regex parsing."""
    if not text:
        return
    lower = text.lower()
    if "cuda error" in lower or "out of memory" in lower:
        raise RuntimeError(
            "gpu_attest_40xx failed on the GPU (see below). This is often caused by "
            "insufficient free VRAM — stop other GPU workloads first (e.g. exit chat.py / LLM "
            "inference), or run attest on an idle GPU via CUDA_VISIBLE_DEVICES=N.\n\n"
            + text.strip()
        )


def resolve_nvcc() -> str:
    """Path to nvcc (CUDA compiler). Uses PATH, then CUDA_HOME / CUDA_PATH, then /usr/local/cuda."""
    w = shutil.which("nvcc")
    if w:
        return w
    for key in ("CUDA_HOME", "CUDA_PATH"):
        base = os.environ.get(key)
        if base:
            cand = Path(base) / "bin" / "nvcc"
            if cand.is_file():
                return str(cand)
    cand = Path("/usr/local/cuda/bin/nvcc")
    if cand.is_file():
        return str(cand)
    raise RuntimeError(
        "Cannot find nvcc (CUDA compiler). Install the CUDA Toolkit, or export PATH so it "
        "includes the toolkit's bin directory (e.g. export PATH=/usr/local/cuda/bin:$PATH), "
        "or set CUDA_HOME to the toolkit root."
    )


def run_cmd(cmd, cwd=ROOT, allow_failure=False):
    p = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    out = (p.stdout or "") + ("\n" + p.stderr if p.stderr else "")
    if p.returncode != 0 and not allow_failure:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{out}")
    return out, p.returncode


def build(cap: str, *, force: bool = False) -> None:
    """Compile gpu_attest_40xx unless the binary is already up to date for this sm_* cap."""
    cap = str(cap)
    if not force and BIN.is_file() and SRC.is_file():
        try:
            meta = json.loads(BUILD_META.read_text(encoding="utf-8"))
            if str(meta.get("cap")) == cap and BIN.stat().st_mtime >= SRC.stat().st_mtime:
                print("[build] skip nvcc (gpu_attest_40xx is up to date for sm_" + cap + ").")
                return
        except (OSError, json.JSONDecodeError, TypeError, AttributeError):
            # Legacy tree: had gpu_attest_40xx before we added .gpu_attest_build.json — do not
            # recompile every time; record cap from this run (must match the built binary).
            if BIN.stat().st_mtime >= SRC.stat().st_mtime:
                print(
                    "[build] skip nvcc (existing binary newer than .cu; "
                    f"recorded sm_{cap} in {BUILD_META.name})."
                )
                try:
                    BUILD_META.write_text(json.dumps({"cap": cap}, indent=2), encoding="utf-8")
                except OSError:
                    pass
                return

    print("[build] running nvcc...")
    nvcc = resolve_nvcc()
    _, code = run_cmd([
        nvcc,
        str(SRC.name),
        f"-arch=sm_{cap}",
        "-O3",
        "-lineinfo",
        "-maxrregcount=64",
        "-o",
        BIN.name,
    ])
    if code != 0:
        raise RuntimeError("Build failed")
    try:
        BUILD_META.write_text(json.dumps({"cap": cap}, indent=2), encoding="utf-8")
    except OSError:
        pass


def parse_runtime(text: str) -> float:
    _raise_if_cuda_program_failed(text)
    m = RUNTIME_RE.search(text)
    if not m:
        raise RuntimeError(f"Cannot parse runtime from output:\n{text}")
    return float(m.group(1))


def parse_gpu_checksum(text: str) -> int:
    _raise_if_cuda_program_failed(text)
    m = GPU_CHK_RE.search(text)
    if not m:
        raise RuntimeError(f"Cannot parse GPU checksum from output:\n{text}")
    return int(m.group(1), 16)


def parse_verify_ok(text: str) -> bool:
    _raise_if_cuda_program_failed(text)
    m = VERIFY_RE.search(text)
    if not m:
        raise RuntimeError(f"Cannot parse verification result from output:\n{text}")
    return m.group(1) == "SUCCEED"


def run_vf_once(args, verify: bool = False, tamper_malicious_benign: int = 0, tamper_malicious_corrupt: bool = False):
    cmd = [
        str(BIN),
        "--iters", str(args.iters),
        "--data-size", str(args.data_size),
        "--grid", str(args.grid),
        "--block", str(args.block),
        "--repeat", str(args.repeat),
    ]

    if tamper_malicious_benign > 0:
        cmd += ["--tamper-malicious-benign", str(tamper_malicious_benign)]
    if tamper_malicious_corrupt:
        cmd += ["--tamper-malicious-corrupt"]

    if verify:
        cmd += ["--verify", "--verify-threads", str(args.verify_threads)]

    out, code = run_cmd(cmd, allow_failure=verify)
    rt = parse_runtime(out)
    chk = parse_gpu_checksum(out)
    v_ok = parse_verify_ok(out) if verify else None
    return rt, chk, v_ok, out


def sample_runtime(args, n: int):
    vals = []
    for _ in range(n):
        rt, _, _, _ = run_vf_once(args, verify=False)
        vals.append(rt)
    return vals


def save_profile(path: Path, d: dict):
    with path.open("w", encoding="utf-8") as f:
        json.dump(d, f, indent=2)


def load_profile(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _profile_path(args) -> Path:
    if getattr(args, "profile", None):
        return Path(args.profile)
    return PROFILE_JSON


def enroll(args):
    profile_path = _profile_path(args)
    print("[enroll] preparing gpu_attest_40xx (nvcc only if needed)...")
    build(args.cap)

    print("[enroll] sampling baseline runtime under full VF execution...")
    baseline = sample_runtime(args, n=args.runs)

    mean = sum(baseline) / len(baseline)
    var = sum((x - mean) ** 2 for x in baseline) / len(baseline)
    sigma = math.sqrt(var)

    margin = sigma * args.threshold_sigma_k

    profile = {
        "gpu_model": args.gpu_model,
        "cap": args.cap,
        "iters": args.iters,
        "data_size": args.data_size,
        "grid": args.grid,
        "block": args.block,
        "repeat": args.repeat,
        "verify_threads": args.verify_threads,
        "baseline_mean": mean,
        "baseline_sigma": sigma,
        "threshold_sigma_k": args.threshold_sigma_k,
        "runtime_upper": mean + margin,
    }
    save_profile(profile_path, profile)

    print("[enroll] profile saved:", profile_path)
    print(f"[enroll] baseline_mean={mean:.6f}s sigma={sigma:.6f}s k={args.threshold_sigma_k:.3f} runtime_upper={profile['runtime_upper']:.6f}s")


def attest(args):
    profile_path = _profile_path(args)
    if not profile_path.exists():
        raise RuntimeError(f"Profile not found: {profile_path}. Run enroll first.")

    profile = load_profile(profile_path)

    print("[attest] preparing gpu_attest_40xx (nvcc only if needed)...")
    build(str(profile["cap"]))

    vt = int(profile.get("verify_threads", 4096))
    if getattr(args, "verify_threads", None) is not None:
        vt = int(args.verify_threads)

    run_args = argparse.Namespace(
        iters=profile["iters"],
        data_size=profile["data_size"],
        grid=profile["grid"],
        block=profile["block"],
        repeat=profile["repeat"],
        verify_threads=vt,
    )

    rt, chk, checksum_ok, out = run_vf_once(
        run_args,
        verify=True,
        tamper_malicious_benign=args.tamper_malicious_benign,
        tamper_malicious_corrupt=args.tamper_malicious_corrupt,
    )

    # Hide verbose Config line for experiments 2/3/4.
    filtered_lines = [ln for ln in out.splitlines() if not ln.startswith("Config:")]
    print("\n".join(filtered_lines).strip())

    time_ok = (rt < float(profile["runtime_upper"]))

    print("\n=== Time Criterion ===")
    print(f"runtime < baseline_mean + k*baseline_sigma")
    print(f"runtime={rt:.6f}s, baseline_mean={float(profile['baseline_mean']):.6f}s, baseline_sigma={float(profile['baseline_sigma']):.6f}s, k={float(profile['threshold_sigma_k']):.3f}")

    print("\n=== Verification Decision ===")
    print(f"checksum_ok={checksum_ok}")
    print(f"time_ok={time_ok}")

    if checksum_ok and time_ok:
        print("verification SUCCEED")
    else:
        print("verification FAILED")
        raise SystemExit(3)


def main():
    p = argparse.ArgumentParser(description="VF attestation where timing covers full checksum execution")
    sub = p.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("enroll", help="Create baseline profile (runtime threshold)")
    pe.add_argument("--gpu-model", default="RTX-40xx")
    pe.add_argument("--cap", default="89")
    pe.add_argument("--runs", type=int, default=30)
    pe.add_argument("--iters", type=int, default=100000)
    pe.add_argument("--data-size", type=int, default=1 << 20)
    pe.add_argument("--grid", type=int, default=0)
    pe.add_argument("--block", type=int, default=0)
    pe.add_argument("--repeat", type=int, default=1)
    pe.add_argument("--threshold-sigma-k", type=float, default=1.0,
                    help="timing threshold multiplier k in runtime < mean + k*sigma")
    pe.add_argument("--verify-threads", type=int, default=4096)
    pe.add_argument(
        "--profile",
        default=None,
        help="Path to attestation_profile.json (default: LSAGE/attestation_profile.json)",
    )
    pe.set_defaults(func=enroll)

    pa = sub.add_parser("attest", help="Run one attestation check using saved profile")
    pa.add_argument("--tamper-malicious-benign", type=int, default=0)
    pa.add_argument("--tamper-malicious-corrupt", action="store_true")
    pa.add_argument(
        "--verify-threads",
        type=int,
        default=None,
        help="Override profile verify_threads (CPU/GPU per-thread subset verification count). Default: from profile.",
    )
    pa.add_argument(
        "--profile",
        default=None,
        help="Path to attestation_profile.json (default: LSAGE/attestation_profile.json)",
    )
    pa.set_defaults(func=attest)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
