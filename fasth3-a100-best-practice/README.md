# FastH3 · 单卡 A100 推理优化 SOTA 最佳实践

> 本目录是本优化方案（**推理方案最佳实践**）的代码面交付：分支 `fasth3-a100-best-practice`
> 描述**方案本身**，不含任何个人 handle；**不开 PR**（内部交付面）。
> 统一启动旋钮见同目录 [`launch.env`](launch.env)（与 `tasks/fasth3/configs/launch.env` 内容一致，单一真相源）。

## 1. 方案一句话

MiniMax **FastH3**（33B dense DiT，4 步 DMD2 蒸馏，T2VA 视频生成）在**单卡 A100（SM80）**上的工程最优：

- **默认采纳 = OPT-1（无损）**：`--layerwise-offload-components dit text_encoder`
  ——只对 DiT 与文本编码器逐层 offload，把仅 ~5 GB、每请求必用的 **VAE 组常驻显存**，
  消除解码阶段逐请求 ~9.8 GB 权重的 PCIe 重流式。
  **e2e 149.75 → 84.96 s（−43.3%）**，输出 mp4 与冻结基线 **MD5 bit-exact**（零质量损失）。
- **代码改动（本分支承载）= VSA-H3 sm80 裸指针 Triton 内核**（capability-gated 双内核）：
  使 FastH3 原生的 90% 稀疏注意力（VSA-H3）在 A100 上真正跑快，
  **e2e 59.59 s（−29.9%）**（有损，PSNR 14.65 dB）；作为**可选快速档**，默认不启用。

> FastH3 权重本就按 VSA-H3 蒸馏（VSA 是模型原生形态，dense `fa` 才是 A100 fallback）；
> 采用 VSA 快速档须按 §6 流程人工抽检后由使用方决定。

## 2. 代码改动清单（本分支相对 `main` 的增量）

本分支基点 = fork 分支 `vsa-h3-a100-sm80-kernel`（tip `32aa25fee0`，即已交付的 VSA 双内核改动；
local base `main` = `62ff466ac0`）。增量 = 2 个 commit、3 个文件、188 行：

| commit | 内容 | 文件 |
| --- | --- | --- |
| `68bd60a270` | VSA-H3 支持 sm80（A100）的裸指针 Triton 加载路径 | `runtime/layers/attention/backends/vsa_h3_kernels.py`、`runtime/platforms/cuda.py`、`test/unit/test_vsa_h3_attention.py` |
| `32aa25fee0` | 把 sm100 tcgen05 内核的 import 改为**惰性**（移进 `use_sm100_kernel` 分支内） | `vsa_h3_kernels.py` |

改动要点：
1. 新增 `_attn_fwd_sparse_ptr`——与 `_attn_fwd_sparse` 相同的算术，但 Q/K/V/O 走**裸指针**而非
   `TensorDescriptor`（TMA）加载；新增 `_use_pointer_loads(capability)`（`capability[0] < 9`）
   与 `_ATTN_NUM_STAGES_PTR = 2`。**sm90+ 行为完全不变**（仍走 TMA，`num_stages=5` 保留）。
   缘由：Triton 只在 Hopper 及更新架构把 `TensorDescriptor` 降级为 `cp.async` 预取；
   在 sm80 描述符加载被 silently 降级为同步 tile 加载、每个 K/V tile 暴露完整延迟 → 内核慢数倍。
2. `_VideoSparseAttentionH3BackendResolver.supported_capabilities` 增补 `(8, 0)`（放行 A100）；
   更新 fail-closed 错误信息。
3. sm100 内核改惰性 import ⇒ 模块在无 sm100 内核的栈上也能 import（该分支在 sm80 永不执行），
   并把这份性质用一个 CPU 单测钉住。
4. 新增 3 个 CPU 单测：`test_pointer_load_path_selection`、`test_vsa_h3_backend_resolver_allows_sm80`、
   `test_vsa_h3_module_imports_without_sm100_symbol`。

> 注：`--layerwise-offload-components dit text_encoder`（OPT-1）是**纯启动配置**，无代码改动，
> 已写入同目录 `launch.env`。

## 3. 效果表（固定测试集：T2VA · 1344×768 · 5 s · 24 fps · 4 去噪步 · seed 1000 · A100 80GB）

| 配置 | text (s) | denoise (s) | decode (s) | **e2e (s)** | 提速 | 峰值显存 (MB) | 质量（vs 冻结基线） |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 初始默认（全组件逐层 offload） | 3.96 | 67.92 | 77.46 | 149.75 | 1.00× | 15740 | 参考（md5 `60e0244b…`） |
| **SOTA = OPT-1（无损）** | 3.97 | 67.35 | 13.26 | **84.96** | **1.76×** | 20210 | **bit-exact**（md5 相同） |
| INT8 kitchen_int8（有损备选） | 3.96 | 59.19 | 13.39 | 76.87 | 1.95× | 20784 | PSNR 27.55 dB（超 §6） |
| **VSA 稀疏（最快档，有损）** | 3.96 | 41.90 | 13.31 | **59.59** | **2.51×** | 23654 | PSNR 14.65 / 音 0.60（超 §6） |

- 内核构成（denoise 窗口，baseline 5 s）：FlashAttention(sm80) **60.8%** / cuBLAS bf16 GEMM **36.9%** /
  norm+act+rope 等已融合算子 2.3%。
- VSA 内核微基准：变体间 bit-exact；**裸指针 vs TMA 描述符 = 7.87×**（PR 复测 7.80×）。
  e2e 内真实内核 `_attn_fwd_sparse` = **57.0 ms/call**（stdev 0.68），无热降频。

## 4. 证据指针（均在 `tasks/fasth3/`，本分支只承载代码面）

- 手册：`docs/fasth3-progress-and-runbook.md` · **核心报告**：`results/FastH3-A100-profile-optimization-report.md`
- OPT-1 采纳：`profiles/20261011-0749_vae-resident-opt/`（e2e 84.96 s，MD5 bit-exact）
- 内核深挖：`results/20261011_kernel_deepdive_denoise.json`、`results/20261011_attention-ncu-finding.md`
- INT8：`results/20261011_quant-int8-a100-finding.md`、`profiles/20261011-1706_exp-q-kitchen-int8/`
- VSA 内核：`results/20261011_vsa-a100-kernel-finding.md`、`results/20261012_vsa-kernel-v5-v6-pipeline-finding.md`、
  `results/20261012_vsa-e2e-trace-finding.md`、`results/20261012_flashinfer-reference-finding.md`
- VSA PR 化 + 部署：`results/20261012_vsa-pr-description.md`、`profiles/20261012-1726_exp-vsa-pr-deployed/`、
  `profiles/20261012-1826_fasth3-service/`
- 补丁（真相源）：`src/patches/0001…0005-*.patch`

## 5. 复现

```bash
# OPT-1（无损 SOTA）一次性 run
bash scripts/orchestrate_remote.sh vae-resident-opt smoke "--layerwise-offload-components dit text_encoder"

# VSA 快速档（有损；内核已在 overlay 内，PR 形态）
bash scripts/orchestrate_remote.sh <tag> smoke "--attention-backend video_sparse_attn_h3 --layerwise-offload-components dit text_encoder"

# 常驻服务（默认无损 OPT-1，端口 8080）
bash scripts/deploy_fasth3_service.sh
```

> 证据边界：单卡 A100 80GB **PCIe**（SM80，无 FP8/NVFP4 张量核、无 VSA 专用 sm100a 内核）；
> 仅 1344×768 / 5 s / 4 步画像。A100 无 FP8/NVFP4；VSA 为 90% 稀疏近似（有损）；
> INT8/VSA 两档均超 `AGENTS.md` §6「产出效果不受影响」门槛，默认配置维持 bit-exact 的 OPT-1。
