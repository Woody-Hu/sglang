# VLM Prefill (Qwen3.5-4B) · 单卡 A100 推理优化 SOTA 最佳实践

> 本目录是本优化方案（**推理方案最佳实践**）的交付面：分支 `vlm-prefill-chunked-best-practice`
> 描述**方案本身**，不含任何个人 handle；**不开 PR**（内部交付面）。
> 统一启动旋钮见同目录 [`launch.env`](launch.env)（与 `tasks/vlm-prefill/configs/launch.env`
> 内容一致，单一真相源）。

## 1. 方案一句话

**Qwen3.5-4B**（32 层混合注意力：8 层 full-attn + 24 层 GDN linear-attn；ViT 24 层，
1 visual token = 32×32 px）在**单卡 A100（SM80）**上，多图 / 大图 / 短文本 VLM 服务画像的最优：

- **采纳项 = 纯配置，无代码改动**：`--chunked-prefill-size 16384 --context-length 16384 --mem-fraction-static 0.80`
  —— 整段 prefill 不切块（请求 ~4.2k token 一次成块）。
  **吞吐 39.16 → 42.39 tok/s（+8.25%）、e2e p50 −7.1%、TPOT p50 −53.1%、TTFT p95 −7.0%**；
  代价 TTFT p50 +8.3%（首块完成更晚）。质量 PASS（mmmu/docvqa/chartqa，bit-exact 处同代码噪声地板）。
- 收益**全部来自 LLM 侧**：更大 prefill chunk = 更少次、更密的 LLM forward（TPOT p50 −53% 即其直接体现）；
  ViT 侧不受影响（三臂 ViT 总时长 45.6/46.7/49.6 s 几乎相同、ms/图恒定 45–49）。

> 本分支基点 = **fork `main` `4560da58ee`，stock 代码**（本方案**不含任何功能补丁**；
> 仅承载一个被动 ViT 指标插桩文件于 `profiles/` raw，不在分支内）。改动面 = **0 行代码**。

## 2. 代码改动清单

| 项 | 内容 |
| --- | --- |
| 代码改动 | **无**（stock fork `main` `4560da58ee`） |
| 采纳面 | 纯启动配置：`--chunked-prefill-size 16384` + `--context-length 16384` + `--mem-fraction-static 0.80` |

> 本方案是**纯配置 SOTA**：所有被考察的代码改动（侧流 ViT 预取 / pinned H2D）均已实测否决
> （见 §4）。唯一采纳项不需要任何源码改动，故本分支落在基点 commit 上，只增加本最佳实践目录。

## 3. 效果表（固定画像：k=4 图 / 1024² / 64 tok 文本 / C=8 / n=256 / ctx 16384 / max_tokens 32）

| 配置 | 吞吐 (tok/s) | TTFT p50 (s) | TTFT p95 (s) | TPOT p50 (s) | e2e p50 (s) | 质量 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 初始默认（chunked 2048） | 39.16 | **3.884** | 4.818 | 0.0405 | 5.211 | 参考 |
| **SOTA（chunked 16384 = ctx）** | **42.39（+8.25%）** | 4.205（+8.3%） | **4.480（−7.0%）** | **0.019（−53.1%）** | **4.841（−7.1%）** | **PASS** |
| ViT CUDA graph（已否决） | 41.49（−2.1% vs SOTA） | 4.343 | 4.561 | 0.0185 | 4.939 | FAIL（mmmu −4pp） |

- 三臂 ViT 总时长 45.6 / 46.7 / 49.6 s、ms/图 45.2 / 46.5 / 49.4 ⇒ 每图 ViT 成本与分组无关，
  采纳项收益**与 ViT 无关**（纯 LLM 侧 chunk 粒度）。
- 质量门（B vs A，`items_multi_k4.jsonl` × 3 集，temp=0，门槛 acc 降 >1pp 即拒）：
  mmmu +2pp（bit_exact 0.94 = 同代码噪声地板）/ docvqa +2pp（瞬态客户端错误内）/ chartqa 0.0 ⇒ **PASS**。

## 4. 被否决的杠杆（4 个，均为实测结论，勿开）

| # | 杠杆 | 形式 | 结论 | 依据 / 原因 |
| --- | --- | --- | --- | --- |
| ① | `--mm-attention-backend flashinfer_cudnn` | 配置 | **REJECT** | +3.1% 吞吐，但 mmmu −4pp（2 题 temp=0 确定性真翻转）超 1pp 门槛；`sdpa` 慢 6×。保留 A100 默认 `triton_attn`（fa3/fa4 需 SM90+ 不可用） |
| ② | HtoD 去 Pageable（pinned staging） | 代码（`SGLANG_MM_PINNED_H2D=1`） | **NULL** | markdown 确证在主链上但零收益。pageable HtoD 只占 encoder 12%、encoder ≈ wall 29% ⇒ 上界 ≤3.5% of wall（＜噪声）；同 stream 串行 bubble 未消 |
| ③ | encoder↔LLM 侧流 ViT 预取（O1） | 代码（`SGLANG_MM_IMAGE_PREFETCH=1`） | **REJECT（有害）** | 单卡共享 GPU 上挪 ViT 不消除工作量、与在飞 LLM forward 争 SM + 与调度争 host；吞吐 **−8.9%**、TTFT p50 **+21.5%**。本画像图像几无复用（unique 1024 ≈ 一对一）⇒ 预取无法把未来 miss 变持久 hit（预取 915 图但准入仍 328 次 ViT，净增 257 批） |
| ⑤ | ViT CUDA graph（`SGLANG_VIT_ENABLE_CUDA_GRAPH=1`） | 配置 | **REJECT** | 吞吐 −2.1%、ms/图 +6.3%（发射开销非瓶颈）；且 **mmmu −4pp**（graph 走 legacy 非向量化 pos-embed 插值，数值不同）+ bit_exact 0.58 ≪ 0.94 地板；memfrac 0.80 首捕 OOM 需降 0.70 |

> 本画像 5 个候选杠杆全部闭环：①REJECT / ②NULL / ③REJECT（有害）/ **④ADOPT（本方案）** / ⑤REJECT。
> 调度/重叠面（①②③⑤）与算子面均无「保质量且有收益」的剩余增量；唯一采纳项为纯配置。

## 5. 证据指针（均在 `tasks/vlm-prefill/`，本分支只承载交付面）

- 手册：`docs/vlm-prefill-progress-and-runbook.md`（§7.12–§7.15 候选杠杆闭环、§9 最终结论 / recipe）
- **采纳项（④）证据**：`profiles/20261017-1558_main-chunked-vitcg-ab/`（`run.json` + `summary.md` + `env.txt`）
- 淘汰项证据：① `profiles/20261017-1202_main-mm-attn-ab/`；② `profiles/20261017-1319_main-pinned-h2d/`；
  ③ `profiles/20261017-1438_main-o1-prefetch-ab/`；⑤ 同 ④ 的 C 臂
- 瓶颈诊断：`results/20261016-2145_p1b-l2v2/roof-model.md`（屋顶模型）、
  `results/20261017-0823_idle-gap/summary.md`（idle 分布）、`profiles/20261017-1033_main-regime-sweep/`（每图 ViT 恒定 40–44 ms）
- 消融表：`results/20261017-0713_knob-sweep/ablation.md`（A0–A5 / B1–B5 / C1–C5 全杠杆 + 判定）
- 补丁存档（均否决，非采纳面）：`src/patches/archive-0011-*`（O1 侧流预取）、`src/patches/archive-0013-*`（ViT 指标插桩）

## 6. 复现

```bash
# SOTA（纯配置）一次性/服务启动（远端宿主 143 · GPU 3）：
shared/scripts/remote_exec.sh \
  'CUDA_VISIBLE_DEVICES=3 CHUNKED_PREFILL_SIZE=16384 CONTEXT_LENGTH=16384 \
   bash /share/home/xueyige/hudi/vlm-prefill/scripts/launch_server_remote.sh'
# 压测：scripts/regime_sweep_remote.sh（k4 / 1024² / 64 tok / C=8 / n=256）
# 停止：docker rm -f vlm-sglang（= scripts/stop_server_remote.sh）
```

> 证据边界：单卡 A100 80GB **PCIe**（SM80，无 FP8/FP4 张量核；fa3/fa4 与 flashinfer/cutedsl GDN
> 内核需 SM90+，A100 上不可用），引擎 = SGLang fork `main`（`4560da58ee`），
> 画像 = k=4 图 / 1024² / 短文本 / C=8。本方案收益集中于 LLM 侧 prefill chunk 粒度，
> 对 k=1 长上下文旧画像与更高图像复用率负载**未验证**。
