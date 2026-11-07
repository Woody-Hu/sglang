# MiniMax-H3 on 4x RTX 5090 (32 GiB) — best practice

Single 4-GPU replica, `fl2va`, TP4, `sp=1`, `performance-mode speed`, monolithic.
Measured 2026-11-07 on 4x RTX 5090 (sm_120), all four cards behind one PCIe Gen4
switch (P2P 25.5 GB/s per card, host uplink 13.4 GB/s aggregate).

## Result

| case | baseline | this recipe | delta |
| --- | --- | --- | --- |
| 512p / 4 steps / 4 s | 21.07 s, 2.845 req/min | **16.07 s, 3.735 req/min** | **−23.7% latency, +31% throughput** |
| 768p / 8 steps / 4 s | 60.69 s, 0.989 req/min | **46.15 s, 1.300 req/min** | **−24.0% latency, +31% throughput** |

Baseline = bf16 transformer + `--vae-cpu-offload --text-encoder-cpu-offload`,
3-4 requests per point, `ok/fail = 2/0` for the recipe rows.

Reference points on the way there (768p / 8 steps):

| configuration | p50 | req/min |
| --- | --- | --- |
| bf16 + TE/VAE cpu-offload (baseline) | 60.69 s | 0.989 |
| bf16 + `text_encoder=snapshot-offload` | 57.18 s | 1.049 |
| bf16 + snapshot + lazy-codec fix | 53.42 s | 1.123 |
| int8 ConvRot, no other change | 53.68 s | 1.118 |
| **int8 + snapshot + lazy-codec fix** | **46.15 s** | **1.300** |

## Recipe

1. **int8 ConvRot transformer** (`--quantization convrot_int8`). Drops per-card
   footprint from ~30.9 GiB to ~26.7 GiB at 768p, which is what gives the
   decoder and the text-encoder any room at all. Contributes ~1.13x on denoise.
2. **`--component-residency text_encoder=snapshot-offload`**. Keeps a pinned host
   snapshot so the per-request 13.2 GiB D2H back to CPU disappears
   (−3.96 s/request at bf16).
3. **`start_at_stage_entry=False` on the H3 reference-encoding stages** (the fix
   in `hudi/h3-vae-double-load`). A text-to-video+audio request no longer pays a
   5.2 GiB H2D+D2H for `video_vae` and 0.56 GiB for `audio_vae` that it never
   uses (−3.5 s at 512p, −3.8 s at 768p).
4. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Without it the 26.7 GiB
   int8 footprint plus warmup activations fragments and OOMs.

## Where the time goes (512p / 4 steps, recipe)

```
TextEncoding 4.25 s   <- 4 x 13.09 GiB of bf16 encoder streamed at 13.4 GB/s
VisualEncoding 0.0 s
AudioEncoding 0.0 s
Denoising 5.9 s       <- the only fully GPU-bound stage
Decoding 3.6 s        <- 26% GPU; the rest is the one legitimate video_vae load
overhead ~1.7 s
```

The text encoder alone is a 63 GiB bf16 checkpoint; H3 taps most of it (52 GiB
materialized), and TP4 shards it to 13.09 GiB per rank. That streaming is the
single largest non-denoise cost and it recurs on every request because the DiT
leaves no room to keep the encoder resident.

## What does not work on this hardware

| attempt | outcome |
| --- | --- |
| TP2 (bf16 or int8) + SP | encoder/DiT do not fit; `tp2 x sp2` OOMs while loading (31.10 GiB used, 498 MiB short) |
| text encoder resident | OOM: DiT + TE exceeds 31.36 GiB by 64 MiB at bf16 |
| VAE resident / VAE layerwise offload | OOM during warmup (`31.28 GiB in use`, 64 MiB short) |
| request-level overlap (decode N ∥ denoise N+1) | needs DiT+VAE co-resident; OOM |
| CUDA graph / `--enable-breakable-cuda-graph` | GPU is already 100% busy inside a step; no launch bubbles to remove (max inter-kernel gap 0.13 ms) |
| VSA-H3 sparse attention | blocked by a capability gate: `VSA_H3_CAPABILITIES = {(8,0),(9,0),(10,0),(10,3)}`, RTX 5090 is `(12,0)` |
| NCCL tuning | all-reduce already at 96% of the Gen4 x16 link; `NCCL_ALGO=Tree` is 25% worse |
| 2 x TP2 instances | text encoder streaming doubles (2 x 52 GiB) onto the same 13.4 GB/s uplink, cancelling the concurrency gain |

## Launch

See `launch.env`.
