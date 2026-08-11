# CoveInfer

**Cove** 将差分隐私（DP）对话推理、zkLLM Layer0 零知识验证、GPU 可信证明（SAGE）与 benchmark 评估整合在同一仓库中。

本仓库名 **CoveInfer**，用于公开发布 / 复现；不含模型权重与运行时中间产物（benchmark 结果、DP 查找表、zkLLM 工作目录等需本地生成）。

## Demo video

演示视频通过 **GitHub Release** 托管（不在 git 仓库内，`git clone` 不会下载）：

<video src="https://github.com/cryptosdu/CoveInfer/releases/download/demo/CoveInfer_English.mp4" controls width="720"></video>

- [cryptosdu 在线观看 / 下载](https://github.com/cryptosdu/CoveInfer/releases/tag/demo)
- [Baipanda 在线观看 / 下载](https://github.com/Baipanda/CoveInfer/releases/tag/demo)

## 入口

| 脚本 | 作用 |
|------|------|
| `chat.py` | Gradio UI：Chat 推理（可选 DP）+ zkLLM Layer0 验证 |
| `evaluation.py` | Gradio / CLI：H20、CSV+4090、CSV+5090 等 benchmark |

## 目录结构

```text
.
├── chat.py                 # Gradio 主界面
├── evaluation.py           # 评估仪表盘 / CLI
├── cove_paths.py           # 路径与旧目录名/Cvee→Cove 迁移辅助
├── cove_ui_theme.py        # UI 主题（COVE_UI_THEME）
├── PRISM/                 # DP 词表（按模型分子目录 + 共享构建脚本）
│   ├── llama-2-7b-hf/      # low_freq_words.txt, nearest_tokens_30.npz
│   └── qwen2.5-7b-instruct/
├── LSAGE/                 # SAGE GPU attestation
│   ├── run_attestation.py
│   ├── gpu_attest_40xx.cu
│   └── attestation_profile.json   # baseline 登记文件（运行时生成或随仓库提供）
├── zk-PIM/                 # zkLLM CUDA 证明（详见 zk-PIM/README.md）
│   └── zkllm-workdir/      # pp / commitment / int 缓存（运行时生成）
├── eval-results/           # benchmark 输出（按模型分子目录）
│   ├── llama-2-7b-hf/
│   └── qwen2.5-7b-instruct/
├── zkllm-chat/             # zkLLM 单次验证工作目录（运行时生成）
├── plotting/               # 论文图/表（按模型分子目录）
│   ├── llama-2-7b-hf/
│   └── qwen2.5-7b-instruct/
└── requirements.txt
```

`offline-wheels-h20/` 等为部署用离线包，已被 Git 忽略。

---

## 安装

需要可用的 CUDA、NVIDIA 驱动与 `nvcc`（zkLLM / SAGE 会编译 CUDA 小程序）：

```bash
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available())"
```

离线安装示例：

```bash
pip install --no-index --find-links /path/to/offline-wheels -r requirements.txt
```

---

## 运行 Chat UI

```bash
python chat.py
```

浏览器打开 `http://localhost:7860`。

可选 UI 主题（旧名 `CVEE_UI_THEME` 仍可用）：

```bash
COVE_UI_THEME=warm_gray python chat.py
```

界面两个标签页：

1. **Chat Inference (DP + Chat)** — 加载模型、可选 DP、生成回复；可选在推理前跑 SAGE。
2. **zkLLM Verification (Layer0)** — 仅跑 zkLLM 验证流水线；可选在验证前跑 SAGE。

默认模型路径为 `/home/data/models/Llama-2-7b-hf`，`cache_dir` 为 `./model-storage`（相对仓库根目录）。**新机器上务必改成实际路径。**

---

## SAGE GPU 证明（`LSAGE/`）

### 做什么

SAGE 在 GPU 上运行固定工作负载 `gpu_attest_40xx`，校验：

1. **checksum**：GPU 与 CPU 子集校验一致（防篡改）。
2. **timing**：单次 `Runtime` 必须低于登记时的阈值（防异常慢/被挂起）。

### 核心文件

| 文件 | 含义 |
|------|------|
| `LSAGE/attestation_profile.json` | 登记结果：`baseline_mean`、`baseline_sigma`、`runtime_upper`、`iters`、`verify_threads` 等 |
| `LSAGE/gpu_attest_40xx` | 编译后的 GPU 可执行文件 |
| `LSAGE/run_attestation.py` | `enroll`（登记）/ `attest`（验证）入口 |

可通过环境变量 `COVE_SAGE_PROFILE`（或旧名 `CVEE_SAGE_PROFILE`）指定 profile 路径；默认即 `LSAGE/attestation_profile.json`。

### `baseline_mean` 与 `runtime` 的区别（重要）

- **`baseline_mean`**：在 **GPU 空闲** 时多次运行 kernel 的平均时间，写在 profile 里（H20 上典型约 **0.35s**，`iters=10000`）。
- **`runtime`**：本次 `attest` 打印的 `Runtime: x.xxx s`，是**这一次**测量的耗时。
- **`runtime_upper`**：`baseline_mean + k × baseline_sigma`（默认 k=2），本次 `runtime` 必须 **严格小于** 它才算 `time_ok=True`。

**不要混淆**：日志里 `runtime=0.35` 是正常的单次测量；`baseline_mean=0.348` 是登记阈值。若看到 `runtime≈0.76` 而 baseline 仍是 0.35，说明 **当时 GPU 不空闲**（例如 zkLLM 的 `commit-param` / `llama-commit` 正在跑），不是 profile 读错路径。

| GPU 状态 | 典型 runtime | `time_ok` |
|----------|--------------|-----------|
| 空闲 | ~0.35s | True |
| 被 zkLLM commit 等占满 | ~0.75–0.80s | False |

**不会**通过放宽 `runtime_upper` 来“骗过”检查；阈值以 profile 为准。

### 登记（enroll）— 只需在每台机器/每种 GPU 环境做一次

在 **GPU 空闲** 时执行（建议先 `nvidia-smi` 确认无 `commit-param` 等进程）：

```bash
cd LSAGE
python run_attestation.py enroll \
  --profile attestation_profile.json \
  --gpu-model H20 \
  --cap 90 \
  --runs 30 \
  --iters 10000 \
  --data-size 1048576 \
  --threshold-sigma-k 2.0 \
  --verify-threads 128
```

输出中的 `[enroll] baseline_mean=...` 应接近 **0.35s**（H20）。若登记时 GPU 很忙，baseline 会偏高或偏低，后续 attest 容易误 FAIL。

### 验证（attest）

```bash
cd LSAGE
python run_attestation.py attest \
  --profile attestation_profile.json \
  --verify-threads 128
```

成功时末尾为：

```text
checksum_ok=True
time_ok=True
verification SUCCEED
```

### Chat 里如何调用 SAGE

- **Chat 标签页**：若勾选 “Run GPU attestation (SAGE) before inference”，在**加载 LLM 之前**调用 `run_sage_gpu_check()`（避免模型占满显存）。
- **zkLLM 标签页**：若勾选 “Run GPU attestation (SAGE) before zkLLM pipeline”，在 zkLLM 流水线之前调用。
- 实现上通过子进程执行 `run_attestation.py attest`，**不会**在测 SAGE 前调用 `zkllm_subprocess_env()`（避免父进程 `import torch` 后占用 GPU，导致 runtime 虚高）。
- 若检测到 GPU 上有 `commit-param` / `llama-commit` 或利用率过高，会**提前报错**并说明原因，而不是静默测出 0.76s 再 FAIL。

### `evaluation.py` 中的 SAGE

评估 UI 提供 **ENROLL GPU BASELINE** 与 **RUN GPU ATTESTATION** 按钮；跑 benchmark 时若勾选 GPU verification，会在每条 benchmark 前调用 `run_sage_attest()`，逻辑与 `LSAGE/run_attestation.py` 相同。

---

## zkLLM Layer0 验证（`zk-PIM/`）

### 依赖的磁盘产物

Layer0 证明二进制需要 `zk-PIM/zkllm-workdir/Llama-2-{7|13}b/` 下已有：

- `*-pp.bin`
- `layer-0-*-commitment.bin`
- `layer-0-*-int.bin`
- `.artifact_manifest.json`（记录对应的 `model_path` / `cache_dir`）

### 点击「START zkLLM VERIFICATION」时发生什么

```text
（可选）SAGE GPU attestation
    ↓
ensure_zkllm_artifacts
    ├─ 若 workdir 已有完整 layer-0 文件且 manifest 匹配 → cache hit，跳过下面两步
    ├─ 若仅因仓库重命名（Cvee→Cove）导致 cache_dir 变了、权重路径未变 → 只更新 manifest，不重新生成
    ├─ 若 GPU 正被 commit-param 等占用 → 拒绝自动 regen，返回 PREP_FAILED + 说明
    └─ 否则自动执行：
         [prepare 1] llama-ppgen.py
         [prepare 2] llama-commit.py    （可能耗时数小时）
    ↓
prompt_to_layer_input.py → layer_input.bin
    ↓
六步 Layer0：rmsnorm → self-attn → skip → rmsnorm → ffn → skip
    ↓
日志含 “Self attention proof successfully verified!” / “SwiGLU proof complete.” 等 → zkLLM PASS
```

**新服务器、workdir 为空时：会自动跑 `llama-ppgen.py` 和 `llama-commit.py`。** 完成后再次点击会 cache hit，直接进入验证。

### 新服务器推荐流程

1. 修改 UI 中的 **Model path**、**cache_dir**。
2. （推荐）在终端先完成准备，避免 Gradio 长时间阻塞：

```bash
cd zk-PIM
python llama-ppgen.py 7 \
  --model_path /path/to/Llama-2-7b-hf \
  --cache_dir /path/to/Cove/model-storage

python llama-commit.py 7 16 \
  --model_path /path/to/Llama-2-7b-hf \
  --cache_dir /path/to/Cove/model-storage
```

3. 再启动 `python chat.py`，在 **zkLLM Verification** 页点击验证。
4. **zkLLM seq_len** 建议 **2048**（须为 2 的幂，与 FFN 证明约束一致）。

### 超时（环境变量）

| 变量 | 默认 | 说明 |
|------|------|------|
| `COVE_ZKLLM_PREP_TIMEOUT_SEC` | 14400（4h） | `llama-ppgen.py`；`0` = 不限制 |
| `COVE_ZKLLM_COMMIT_TIMEOUT_SEC` | 86400（24h） | `llama-commit.py`；`0` = 不限制 |

### 仓库 / 目录重命名

若仅仓库目录从 `Cvee` 改为 `Cove`/`CoveInfer`，`model_path` 未变且 `zkllm-workdir` 内文件齐全，**不会**因 manifest 里旧的 `cache_dir` 而触发数小时的 commit 重建；会更新 manifest 后直接使用缓存。

组件目录亦已更名（旧路径会由 `cove_paths` 自动改写）：

| 旧目录 | 新目录 |
|--------|--------|
| `dp-sanitization/` | `PRISM/` |
| `sage-main/` | `LSAGE/` |
| `zkllm/` | `zk-PIM/` |

说明：`zkllm-chat/` 与 `zk-PIM/zkllm-workdir/` 未改名。Python 中仍可 `import zkllm`（指向 `zk-PIM/`）。

若确实更换了模型权重路径，则会重新 `ppgen` + `commit`。

### zkLLM 与 SAGE 同时跑时注意

`llama-commit` 会启动大量 `commit-param` 进程并占满 GPU。在 commit 未完成时：

- SAGE 的 `runtime` 会升到 ~0.76s，`time_ok=False` → **SAGE FAIL**（预期行为）。
- UI 会拒绝**自动开始**新的 pp/commit 重建（若 GPU 已忙）。

请先 `nvidia-smi` 确认 GPU 空闲，或等 commit 结束后再做 SAGE / Chat。

---

## 运行 Evaluation

Gradio 仪表盘：

```bash
COVE_FORCE_MATH_SDP=1 python evaluation.py --ui
```

单条 H20 CLI 示例：

```bash
COVE_FORCE_MATH_SDP=1 python evaluation.py \
  --model_ref /home/data/models/Llama-2-7b-hf \
  --config H20 \
  --device cuda:0 \
  --dtype fp16
```

结果写入 `eval-results/<model-slug>/`（`.csv` / `.jsonl`），例如 `eval-results/llama-2-7b-hf/`、`eval-results/qwen2.5-7b-instruct/`。

---

## 环境变量汇总

| 变量 | 用途 |
|------|------|
| `COVE_UI_THEME` | Gradio 主题：`default`、`warm_gray`、`ivory`、`soft_blue`、`sage` |
| `COVE_FORCE_MATH_SDP` | 强制 math-only SDPA（H20/Hopper 稳定）；旧名：`ZKLLM_FORCE_MATH_SDP`、`CVEE_*`、`CVINF_*` |
| `COVE_ZKLLM_PYTHON` | zkLLM 子进程 Python 路径 |
| `COVE_ZKLLM_PREP_TIMEOUT_SEC` | `llama-ppgen.py` 超时（秒）；`0` = 不限 |
| `COVE_ZKLLM_COMMIT_TIMEOUT_SEC` | `llama-commit.py` 超时（秒）；`0` = 不限 |
| `COVE_SAGE_PROFILE` | `attestation_profile.json` 路径 |
| `COVE_SAGE_ENROLL_RUNS` | SAGE enroll 采样次数（默认 `30`） |
| `COVE_SAGE_ENROLL_ITERS` | SAGE enroll / attest 的 kernel iters（默认 `10000`） |
| `COVE_SAGE_VERIFY_THREADS` | SAGE attest 的 `--verify-threads`（默认 `128`） |

旧项目名环境变量（`CVEE_*`、`CVINF_*`）在代码中仍作为回退读取，**新配置请使用 `COVE_*`**。

---

## 运行时生成的文件

| 路径 | 说明 |
|------|------|
| `zk-PIM/zkllm-workdir/Llama-2-*b/` | pp、commitment、int 与 manifest |
| `zkllm-chat/<时间>_<prompt-slug>/` | 单次验证的中间 `.bin`、`prompt.txt`、`run_meta.json` |
| `zk-PIM/` 下编译产物 | `rmsnorm`、`ffn`、`ppgen` 等（`make` 生成） |
| `LSAGE/attestation_profile.json` | SAGE baseline（enroll 写入） |
| `PRISM/<model-slug>/` | 各模型 DP 词表（`low_freq_words.txt`、`nearest_tokens_30.npz`） |
| `eval-results/<model-slug>/` | benchmark 输出 |
| `plotting/<model-slug>/` | 论文图与 LaTeX 表 |

均已列入 `.gitignore`（除你主动提交的 profile 示例外）。

---

## 常见问题

### SAGE：checksum 通过但 status 仍是 FAIL

看日志里 `time_ok=False` 且 `runtime` 明显大于 `baseline_mean`（例如 0.76 vs 0.35）。处理：结束占用 GPU 的进程（`pkill -f commit-param` 等），`nvidia-smi` 确认空闲后重试；不要在 commit 跑满时测 SAGE。

### SAGE：baseline 应该是多少？

在 **空闲 H20**、`iters=10000`、`verify_threads=128` 下，登记结果 `baseline_mean` 约 **0.34–0.35s**。应用 **0.76** 作为 baseline 通常是登记时 GPU 不空闲或搞混了 `runtime` 与 `baseline_mean`。

### zkLLM：第一次点击要等很久

正常。`llama-commit.py` 可能对单层权重跑很久。可先在终端跑完 ppgen/commit，再在 UI 验证。

### 从 Cvee 改名 Cove 后 zkLLM / SAGE 异常

检查 `zkllm-workdir/.../.artifact_manifest.json` 中的 `cache_dir` 是否已指向 `Cove`；manifest 与权重未变时不应触发全量 commit。若已在跑 commit，先等其结束或结束进程再测 SAGE。

### Chat 报 `GPU status: ERROR` / `torch` 未定义

请使用最新 `chat.py`（推理前会 `torch = _torch()` 延迟加载）。重启 UI 后再试。

---

## 相关文档

- zkLLM 技术细节：`zk-PIM/README.md`
- 绘图脚本：`plotting/`（见 `plotting/README.md`；输出到 `plotting/<model-slug>/`）
