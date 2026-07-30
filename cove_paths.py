"""
Shared repo paths and legacy CVEE/Cvee/CVINF → Cove migration helpers.

Used by chat.py, evaluation.py, cove_ui_theme.py, and zkLLM artifact manifests.

On-disk component dirs (current):
  PRISM/   (was dp-sanitization/)
  LSAGE/   (was sage-main/)
  zk-PIM/  (was zkllm/; Python import alias remains ``zkllm``)
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
from typing import Any, Iterable, Optional

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

# Current on-disk directory names
PRISM_DIR_NAME = "PRISM"
LSAGE_DIR_NAME = "LSAGE"
ZK_PIM_DIR_NAME = "zk-PIM"

ZKLLM_CHAT_DIR = os.path.join(REPO_ROOT, "zkllm-chat")
PRISM_DIR = os.path.join(REPO_ROOT, PRISM_DIR_NAME)
DP_SANITIZATION_DIR = PRISM_DIR  # backward-compatible alias
LSAGE_DIR = os.path.join(REPO_ROOT, LSAGE_DIR_NAME)
ZK_PIM_DIR = os.path.join(REPO_ROOT, ZK_PIM_DIR_NAME)
EVAL_RESULTS_DIR = os.path.join(REPO_ROOT, "eval-results")
PLOTTING_DIR = os.path.join(REPO_ROOT, "plotting")

DEFAULT_LLAMA_MODEL_REF = "/home/data/models/Llama-2-7b-hf"
DEFAULT_QWEN_MODEL_REF = "/home/data/models/Qwen2.5-7B-Instruct"

_MODEL_SLUG_RE = re.compile(r"[^a-zA-Z0-9._+-]+")

# Directory names used before the repo was renamed to Cove.
LEGACY_REPO_DIR_NAMES = ("Cvee", "cvee", "CVEE", "CVee", "CVINF", "cvinf")

# Component directory renames (old → new). Applied after repo-name migration.
# ``zkllm`` must not match ``zkllm-chat`` / ``zkllm-workdir`` (boundary-aware rewrite).
LEGACY_COMPONENT_DIR_MAP = (
    ("dp-sanitization", PRISM_DIR_NAME),
    ("sage-main", LSAGE_DIR_NAME),
    ("zkllm", ZK_PIM_DIR_NAME),
)

# Path-segment boundary: old name as a full path component only.
_COMPONENT_SEG_RE = {
    old: re.compile(rf"(^|[/\\]){re.escape(old)}(?=[/\\]|$)")
    for old, _ in LEGACY_COMPONENT_DIR_MAP
}


def env_str(primary: str, *legacy: str) -> str:
    """Read primary env var, then optional legacy names (CVEE_*, CVINF_*, ZKLLM_*, …)."""
    for key in (primary, *legacy):
        val = os.environ.get(key, "")
        if val is not None and str(val).strip():
            return str(val).strip()
    return ""


def repo_path(path: str) -> str:
    if not path or os.path.isabs(path):
        return path
    return os.path.join(REPO_ROOT, path)


def model_slug(model_ref: str) -> str:
    base = os.path.basename(model_ref.rstrip("/"))
    slug = _MODEL_SLUG_RE.sub("-", base).strip("-").lower()
    return slug or "unknown-model"


def resolve_model_subdir(base_dir: str, model_ref: str) -> str:
    slug = model_slug(model_ref)
    norm = os.path.normpath(repo_path(base_dir))
    if os.path.basename(norm) == slug:
        return norm
    return os.path.join(norm, slug)


def resolve_results_dir(model_ref: str, results_dir: str) -> str:
    """Benchmark JSONL/CSV under eval-results/<model-slug>/."""
    return resolve_model_subdir(results_dir, model_ref)


def dp_tables_dir(model_ref: str) -> str:
    """DP lookup tables under PRISM/<model-slug>/."""
    return resolve_model_subdir(PRISM_DIR, model_ref)


def dp_low_freq_words_path(model_ref: str) -> str:
    return os.path.join(dp_tables_dir(model_ref), "low_freq_words.txt")


def dp_nearest_tokens_npz_path(model_ref: str) -> str:
    return os.path.join(dp_tables_dir(model_ref), "nearest_tokens_30.npz")


def dp_low_freq_words_rel(model_ref: str) -> str:
    return os.path.relpath(dp_low_freq_words_path(model_ref), REPO_ROOT)


def dp_nearest_tokens_npz_rel(model_ref: str) -> str:
    return os.path.relpath(dp_nearest_tokens_npz_path(model_ref), REPO_ROOT)


def plotting_out_dir(model_ref: str) -> str:
    """Paper figures under plotting/<model-slug>/."""
    return resolve_model_subdir(PLOTTING_DIR, model_ref)


def infer_slug_from_results_dir(results_dir: str) -> Optional[str]:
    norm = os.path.normpath(repo_path(results_dir))
    if os.path.basename(os.path.dirname(norm)) == os.path.basename(EVAL_RESULTS_DIR):
        slug = os.path.basename(norm)
        if slug and slug != os.path.basename(EVAL_RESULTS_DIR):
            return slug
    return None


def default_plotting_out_dir(results_dir: str, model_ref: Optional[str] = None) -> str:
    slug = model_slug(model_ref) if model_ref else infer_slug_from_results_dir(results_dir)
    if slug:
        return os.path.join(PLOTTING_DIR, slug)
    return os.path.join(PLOTTING_DIR, "out")


def canonical_path(path: str) -> str:
    """Stable absolute path (resolve symlinks when the path exists)."""
    p = os.path.normpath(os.path.abspath(repo_path(path)))
    try:
        if os.path.exists(p):
            return os.path.normpath(os.path.realpath(p))
    except OSError:
        pass
    return p


def _rewrite_legacy_component_dirs(path: str) -> str:
    """Rewrite old component directory segments (dp-sanitization, sage-main, zkllm)."""
    out = path
    for old, new in LEGACY_COMPONENT_DIR_MAP:
        out = _COMPONENT_SEG_RE[old].sub(rf"\1{new}", out)
    return out


def normalize_legacy_repo_path(path: str) -> str:
    """Map absolute paths under an old repo folder name to the current REPO_ROOT name."""
    p = canonical_path(path)
    repo_name = os.path.basename(canonical_path(REPO_ROOT))
    for old in LEGACY_REPO_DIR_NAMES:
        if old == repo_name:
            continue
        needle = os.sep + old + os.sep
        if needle in p:
            p = p.replace(needle, os.sep + repo_name + os.sep)
        elif p.endswith(os.sep + old):
            p = p[: -len(old)] + repo_name
        fwd = "/" + old + "/"
        if fwd in p.replace(os.sep, "/"):
            p = p.replace(fwd, "/" + repo_name + "/")
    return _rewrite_legacy_component_dirs(p)


def paths_equivalent(a: str, b: str, *, basename_only: bool = False) -> bool:
    sa = normalize_legacy_repo_path(a)
    sb = canonical_path(b)
    if sa == sb:
        return True
    if basename_only and os.path.basename(sa) == os.path.basename(sb):
        return True
    try:
        root = canonical_path(REPO_ROOT)
        if os.path.commonpath([sa, root]) == root and os.path.commonpath([sb, root]) == root:
            if os.path.relpath(sa, root) == os.path.relpath(sb, root):
                return True
    except ValueError:
        pass
    if os.path.basename(sa) == os.path.basename(sb) == "model-storage":
        return True
    return False


def rewrite_legacy_paths_in_text(text: str) -> str:
    """Replace legacy repo / component directory segments in user-visible logs."""
    if not text:
        return text
    out = text
    repo_name = os.path.basename(canonical_path(REPO_ROOT))
    for old in LEGACY_REPO_DIR_NAMES:
        if old == repo_name:
            continue
        out = out.replace(os.sep + old + os.sep, os.sep + repo_name + os.sep)
        out = out.replace("/" + old + "/", "/" + repo_name + "/")
        if out.endswith(os.sep + old):
            out = out[: -len(old)] + repo_name
        if out.endswith("/" + old):
            out = out[: -len(old)] + repo_name
    return _rewrite_legacy_component_dirs(out)


def _migrate_json_obj(obj: Any) -> tuple[Any, bool]:
    changed = False
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            nv, ch = _migrate_json_obj(v)
            out[k] = nv
            changed = changed or ch
        return out, changed
    if isinstance(obj, list):
        out = []
        for v in obj:
            nv, ch = _migrate_json_obj(v)
            out.append(nv)
            changed = changed or ch
        return out, changed
    if isinstance(obj, str):
        nv = normalize_legacy_repo_path(obj) if (os.sep in obj or "/" in obj) else obj
        if nv != obj:
            return nv, True
        rw = rewrite_legacy_paths_in_text(obj)
        if rw != obj:
            return rw, True
        return obj, False
    return obj, False


def zkllm_chat_dir() -> str:
    """Per-run zkLLM verification workspaces (prompt.txt, layer .bin, run_meta.json)."""
    override = env_str("COVE_ZKLLM_CHAT_DIR", "CVEE_ZKLLM_CHAT_DIR")
    if override:
        return normalize_legacy_repo_path(override)
    return ZKLLM_CHAT_DIR


def sage_profile_path(repo_root: Optional[str] = None) -> str:
    """
    SAGE attestation profile (baseline_mean, runtime_upper, iters, …).
    Override with COVE_SAGE_PROFILE; legacy CVEE_SAGE_PROFILE still accepted.
    """
    root = repo_root or REPO_ROOT
    override = env_str("COVE_SAGE_PROFILE", "CVEE_SAGE_PROFILE")
    if override:
        return normalize_legacy_repo_path(override)
    return os.path.join(root, LSAGE_DIR_NAME, "attestation_profile.json")


def sage_enroll_runs() -> int:
    raw = env_str("COVE_SAGE_ENROLL_RUNS", "CVEE_SAGE_ENROLL_RUNS")
    return int(raw) if raw else 30


def sage_enroll_iters() -> int:
    raw = env_str("COVE_SAGE_ENROLL_ITERS", "CVEE_SAGE_ENROLL_ITERS")
    return int(raw) if raw else 10000


def sage_verify_threads() -> int:
    raw = env_str("COVE_SAGE_VERIFY_THREADS", "CVEE_SAGE_VERIFY_THREADS")
    return int(raw) if raw else 128


def migrate_legacy_repo_paths_in_tree(
    root: str,
    *,
    suffixes: Iterable[str] = (".json",),
) -> int:
    """Rewrite legacy absolute paths inside JSON files under root. Returns files updated."""
    updated = 0
    if not os.path.isdir(root):
        return 0
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if suffixes and not any(name.endswith(s) for s in suffixes):
                continue
            path = os.path.join(dirpath, name)
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                continue
            new_data, changed = _migrate_json_obj(data)
            if not changed:
                continue
            with open(path, "w", encoding="utf-8") as f:
                json.dump(new_data, f, ensure_ascii=False, indent=2)
                f.write("\n")
            updated += 1
    return updated


def register_zkllm_import_alias() -> None:
    """
    ``zk-PIM`` is not a valid Python package name. Register import alias ``zkllm``
    so ``from zkllm.model_load_utils import …`` keeps working.
    """
    alias = "zkllm"
    existing = sys.modules.get(alias)
    if existing is not None and getattr(existing, "__path__", None):
        return
    pkg_dir = ZK_PIM_DIR
    init_py = os.path.join(pkg_dir, "__init__.py")
    if not os.path.isdir(pkg_dir) or not os.path.isfile(init_py):
        return
    spec = importlib.util.spec_from_file_location(
        alias,
        init_py,
        submodule_search_locations=[pkg_dir],
    )
    if spec is None or spec.loader is None:
        return
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)


register_zkllm_import_alias()
