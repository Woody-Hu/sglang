# Qwen3-Next-80B-A3B · 单卡 A100 · 最佳实践（branch `qwen3next-gguf-a100-best`）

> **本分支用途**：把本次优化（`tasks/qwen3next-a100`）的**代码改动 + 统一旋钮调参**固化为一个
> 自包含的最优（SOTA）交付面。
> **不开 PR**（内部自用）；上游 PR 形态另见 `tasks/qwen3next-a100/src/patches/0001–0005`。
>
> 基点：fork `main` = `4560da58ee`；本分支 = 基点 + 全部已采纳改动（P1–P3 + PR-E）。

---

## 1. 一句话结论

单卡 A100（SM80，80 GB）上把 **Qwen3-Next-80B-A3B（hybrid GDN+MTP MoE，3B active）**
以 **GGUF Q4_K_M** 跑通并服务化；核心收益来自 **GGUF MoE 的批量 MMQ 核**
（prefill / TTFT 主导），配 bf16 MTP overlay 无损投机解码（低并发）与一组稳定旋钮。

- **吞吐 SOTA（MMQ，MTP off）**：C=1/8/32 = **72.9 / 174.0 / 347.2 tok/s**（vec 基线 **+76 / +94 / +52%**）；bs=1 TPOT **7.3 ms**
- **质量（Q4_K_M）**：MMLU-Pro 200 CoT **77.0**（官方 80.6，Δ−3.6pp ≈ 1σ）/ MMLU 100 **85.0** / GSM8K 100 **98.0** / HumanEval 164 **98.78**
- **显存**：权重 45.24 GB + pool，memfrac 0.85 下 max_total 495195 tok / max_req 55

## 2. 统一旋钮（`launch.env`，本目录）

`launch.env` 是**唯一旋钮真相源**（与 `tasks/qwen3next-a100/configs/launch.env` 一致）。要点：

| 旋钮 | 值 | 理由 |
| --- | --- | --- |
| `--load-format` | `gguf` | 48.4 GB Q4_K_M 文件直接加载，显存内 4bit |
| `--mem-fraction-static` | `0.85` | 48.4 GB 权重 + pool ≈ 60–65 GB，80 GB 内留余量 |
| `--context-length` | `32768` | 单请求 KV ≈ 0.75 GB，pool 有余量 |
| `--mamba-ssm-dtype` | `bfloat16` | GDN recurrent state 省一半显存 |
| MMQ 批量 MoE | 启用 | `.so` = `/work/scratch/mmq_ext/hudi_gguf_moe.so`（缺则回退 vec） |
| MTP/NEXTN | **C≤8 开 / C=32 关** | 低并发 +22%；高并发有效并对齐后 ≈−11%（draft 4.23 GB 压 cap 55→24） |
| 容器 env | `SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1` | fork 源码 vs 已装 wheel 版本告警 |
| GPU | 仅 `CUDA_VISIBLE_DEVICES=3` | GPU 0 他人占用，禁用 |

启动耗时（P1e 实测）：load ~114 s + prefill CUDA graph ~364 s + decode ~7 s ≈ **8–10 min**。

补丁同步（4 个 srt 文件，见 `launch.env` 的 `PATCH*_SRC`）：

```bash
WT="tasks/qwen3next-a100/vendor/sglang-worktrees/qwen3next-best/python/sglang/srt"
scp "$WT/model_loader/gguf_name_maps.py" "$R:/tmp/q3n_gguf_name_maps.py"
scp "$WT/models/qwen3_next.py"           "$R:/tmp/q3n_qwen3_next.py"
scp "$WT/layers/quantization/gguf.py"    "$R:/tmp/q3n_gguf_quant.py"
scp "$WT/models/qwen3_next_mtp.py"       "$R:/tmp/q3n_qwen3_next_mtp.py"
```

## 3. 代码改动清单（本分支 vs 基点 `4560da58ee`）

| 文件 | 改动 | 对应 PR |
| --- | --- | --- |
| `model_loader/gguf_name_maps.py` | `build_qwen3_next_name_map`（48 层 × GDN 后缀 + MoE shared-expert）+ 单测 7 例 | PR-A |
| `models/qwen3_next.py` | GGUF 感知 GDN packed loader；`ssm_in` 去 qkvz 置换、norm/A_log 逆变换；`get_embed(_and_head)` GGUF-safe | PR-B / PR-E |
| `layers/quantization/gguf.py` | `GGUFMoEMethod` 初始化 + `materialize_gguf_weights()`；sgl_kernel 0.4.7 API 适配；MMQ 批量开关 + `get_gguf_dequantized_weight` | PR-B / PR-D / PR-E |
| `models/qwen3_next_mtp.py` | bf16 `mtp.*` overlay（`SGLANG_GGUF_MTP_OVERLAY`） | PR-C |
| `kernels/aot/csrc/quantization/gguf/moe.cuh` | 专家上限 255→65535 + 修潜伏越界（守卫前置 + `>=`） | PR-D（内核） |
| `test/.../test_qwen3_next_gguf_name_map.py` | name-map 单测 | PR-A |

> MMQ 的真实批量核在**独立扩展** `hudi_gguf_moe`（`tasks/qwen3next-a100/src/mmq_patch/`），
> srt 侧只是开关；不改动已装 `sgl_kernel`。

## 4. 构建 MMQ 扩展（首次 / 内核改动后）

```bash
docker exec q3n-sglang env AOT_SRC=/work/scratch/mmq_src BUILD_DIR=/work/scratch/mmq_ext \
  python3 /work/scratch/mmq_src/build_hudi_gguf.py          # → hudi-mmq-0.4.7-exp65535
docker exec q3n-sglang python3 /work/scratch/mmq_src/test_mmq_patch.py   # → ALL KERNEL TESTS PASSED
```
`.so` 落在 `/work` 挂载 ⇒ 跨容器持久，重启无需重建。

## 5. 效果对比（vs 初始默认 = vec 路径）

| 配置 | C=1 | C=8 | C=32 | bs=1 TPOT |
| --- | --- | --- | --- | --- |
| 初始默认（GGUF vec MoE，PR A+B） | 41.3 tok/s | 89.8 | 228.5 | 7.3 ms |
| **SOTA（+ MMQ 批量）** | **72.9 (+76%)** | **174.0 (+94%)** | **347.2 (+52%)** | 7.3 ms |
| + MTP NEXTN（steps=3/topk=1/draft=4） | 89.2 (+22% vs C=1) | 155.5 | 184.6（cap 24） | 4.7 ms |

TTFT p50（MMQ）：C=1 **0.817 s**（vec 2.156）/ C=8 2.150 s / C=32 0.518 s。

## 6. 证据与真相源

- 完整结论 / 里程碑：`tasks/qwen3next-a100/docs/qwen3next-a100-progress-and-runbook.md`
- 配方 + 14 条坑：`tasks/qwen3next-a100/docs/qwen3next-a100-serving-recipe.md`
- 上游 PR 材料：`tasks/qwen3next-a100/docs/qwen3next-a100-upstream-pr.md`
- 数据：`tasks/qwen3next-a100/profiles/`（baseline / MMQ A/B / MTP / profile）+ `results/`
- 停止方式：`docker rm -f q3n-sglang`（GPU 3 即释放）
