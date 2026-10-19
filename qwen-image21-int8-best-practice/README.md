# Qwen-Image 2.1 · 单卡 A100 推理优化 SOTA 最佳实践

> 本目录是本优化方案（**推理方案最佳实践**）的代码面交付：分支 `qwen-image21-int8-best-practice`
> 描述**方案本身**（INT8 ConvRot + INT8 QKV 打包），不含任何个人 handle；**不开 PR**。
> 统一启动旋钮见同目录 [`launch.env`](launch.env)（单一真相源）。

## 1. 方案一句话

在 A100（SM80，无 FP8/FP4）上对 Qwen-Image 2.1 的 DiT 线性层做 **INT8 ConvRot 权重量化**
（`--component-quantizations.transformer kitchen_int8`），并把注意力 Q/K/V 三个 `[C,C]` 线性
**打包成一次 `[3C,C]` `int8_linear`**，消除每 block 两次冗余 ConvRot 激活量化：

- **denoise 12.539 → 8.843 s（1.41×）**，e2e 12.734 → 9.033 s，峰值显存 23902 → 16574 MB；
- 质量 **PSNR 38.15 dB / SSIM 0.989 → PASS**（lossy 门槛）；QKV 打包为 **bit-exact**（PNG md5 不变）。

## 2. 代码改动清单（本分支相对 `main` 的增量 = patch 0008）

| 补丁 | 基点 | 内容 | 角色 |
| --- | --- | --- | --- |
| `0006-qwen-image21-qkv-pack-offload-guard.patch` | v0.5.21 tag | backport 上游 PR #43576（`packed_qkv_weight` 对 layerwise-offload 的 1-D 占位权重做形状守卫）| 仅 **v0.5.21 镜像**需要；main 已内置 |
| `0007-qwen-image21-int8-packed-qkv.patch` | v0.5.21 tag | INT8 QKV 打包（`kitchen_int8` 版）：`int8_linear` 提为模块级共享入口 + 独立 `_QKV_PACK_INT8_FUSION` 门 + 4 单测 | 容器复现形；**须在 0006 之后** |
| `0008-qwen-image21-convrot-int8-packed-qkv.patch` | `main` | 0007 的上游 PR 形（main 已 rename `kitchen_int8.py`→`convrot_int8_comfy_kitchen.py`）| **本分支已包含** |

本分支 `qwen-image21-int8-best-practice` 的基点 = `hudi/qwen21-int8-pack-main`（tip `a3c1170194`，
`main` 基点 `684fea8af3`），即**已含 0008**；0006/0008 逻辑等价（同一 ConvRot 融合内核 + 同一打包思路，
只是落位文件/类名随基点不同）。

改动要点：
1. 量化后端 `apply` 退化为薄适配层，融合内核提为模块级 `int8_linear(x, weight, weight_scale, group_size, bias=None, split_out_features=None)`
   —— 算子/实参/顺序不变 ⇒ 打包与分离调用同一实现，数值逐比特一致。
2. `qwen_image21.py` 让 QKV 打包覆盖 convrot_int8：int8 `[3C,C]` 权重 + 对应 `[3C,1]` fp32 行 scale 各共享一份存储，
   `_project_qkv_packed_convrot_int8` 用一次 `int8_linear` 出 QKV；`_QKV_PACK_CONVROT_INT8_FUSION` 首次调用对拍 eager 三路，不一致自动回退。
3. `split_out_features` 让 row-split 启发式按**单投影宽度**（而非打包后宽度）判定；K/V 以普通视图返回，避免额外 copy。

> **不打包 FFN gate/proj**：真实情形是 `2×N=12288 → 1×N=24576` 的宽 N 融合，M 扫描显示更慢，省下的 1 次 ConvRot 不足以抵消 GEMM 变慢。

## 3. 效果表（固定测试集：1024×1024 · 40 步 · guidance 1 · seed 42 · A100）

| 配置 | denoise (s) | e2e (s) | 加速 | 峰值显存 (MB) | 质量 |
| --- | ---: | ---: | ---: | ---: | --- |
| 初始默认（bf16 / fa） | 12.539 | 12.734 | 1.00× | 23902 | 参考（md5 `22065bb8…`） |
| INT8 ConvRot（kitchen_int8） | 9.059 | 9.249 | 1.377× | 16228 | PSNR 38.15 / SSIM 0.989 |
| **SOTA = INT8 + QKV 打包** | **8.843** | **9.033** | **1.410×** | 16574 | **bit-exact**（md5 `05288a6…`） |

INT8 内核构成（L2 trace，GPU busy 98.6%）：CUTLASS IMMA GEMM 55.5% / FA2 attention 21.0% / ConvRot 激活量化 13.3%。
INT8 GEMM-only 实测 **452.9 TOPS = 72.6% MFU**（A100 INT8 屋顶 624 TOPS）。

## 4. 证据指针（均在 `tasks/qwenimage21/`，本分支只承载代码面）

- 手册：`docs/qwenimage21-progress-and-runbook.md`
- 优化汇总：`results/20261013_qwenimage21-optimization.md`
- L1/L2/L3：`profiles/20261013-0908_baseline/`、`profiles/20261013-0930_profile/`
- INT8 采纳：#1 `profiles/20261013-1020_exp-kitchen-int8/`；#1b `profiles/20261013-1605_int8-packed-qkv/`(+对照 `…-1625_…control/`)
- INT8 L2 trace：`profiles/20261013-1500_int8-l2-trace/`
- 微基准：`profiles/20261013-1510_int8-gemm-bench/`、`profiles/20261013-1610_int8-qkv-pack-bench/`
- main 形 e2e：`profiles/20261014-1305_int8-packed-qkv-main/`
- 质量脚本：`src/analysis/quality_image.py`（`--gate lossy`）· 结构/MFU：`src/analysis/qwenimage21_structure.py`

## 5. 复现

```bash
# SOTA（main 基点镜像，代码改动已打进镜像）
IMAGE=hudi/sglang:qi21-main-cu130 \
EXTRA_SGLANG_ARGS="--component-quantizations.transformer kitchen_int8" \
  bash scripts/run_qwenimage21.sh <run_dir_rel>

# 备选（v0.5.21 发布镜像内按序 apply 0006 → 0007）
IMAGE=m.daocloud.io/docker.io/lmsysorg/sglang:v0.5.21-cu130 \
PIP_PREINSTALL="comfy-kitchen==0.2.37" \
APPLY_SGLANG_PATCHES="/work/src/patches/0006-qwen-image21-qkv-pack-offload-guard.patch /work/src/patches/0007-qwen-image21-int8-packed-qkv.patch" \
EXTRA_SGLANG_ARGS="--component-quantizations.transformer kitchen_int8" \
  bash scripts/run_qwenimage21.sh <run_dir_rel>
```

> 证据边界：单卡 A100 80GB PCIe（SM80，无 FP8/NVFP4 张量核）；仅 1024×1024 / 40 步画像；
> W4A8/W4A4/FP8 在 A100 + 现有 BF16 checkpoint 下不可用；Turbo 8 步属换模型模式，未纳入本 SOTA。
