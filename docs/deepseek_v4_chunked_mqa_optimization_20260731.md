# DeepSeek V4 chunked mega MQA: single-request audit and optimization ledger

Date: 2026-07-31

## Scope and correctness contract

The canonical prefill cases model one request with `m` causal query tokens, not
`m` independent one-token requests:

- `q`: `[1, m, 64, 128]`
- request lengths: `[1, m]`
- physical page table: `[1, 256]`
- context: `65536`
- formal `m`: `1024`, `2048`, `4096`
- tolerance: `rtol=1e-5`, `atol=1e-6`
- the physical FP8 KV cache is an input and must remain bitwise unchanged

All 20 reference `spec.py` files were inspected. Five references required an
explicit request/token-semantics correction; the other fifteen already had the
correct interpretation. Full correctness after the fixes was 9/9 for paged
MQA, 18/18 for top-k transform, 15/15 for indexer quantization, and 9/9 for the
mega operator. The detailed inventory is in
`docs/reference_casespec_request_semantics_audit.md`.

## Validated optimization points

All timings below are for the canonical single-request suite. NCU values are
from `--set full`, `m=4096`, on GPU 3 under the repository lock.

| Version | Structural change | Formal correctness | m4096 benchmark | NCU duration | TMA traffic | L2 sectors | Registers | Dynamic SMEM | Result |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| V0 | stock control | 9/9 | ~1.57 ms | 1570.14 us | 6.644 GB | 207.802 M | 168 | 220.672 KB | baseline |
| V2 | BLOCK_Q4, two TMEM stages | 9/9 | ~0.98 ms | 950.62 us | 2.202 GB | 287.504 M | high/spilling | 220 KB class | rejected: severe spill |
| V3 | shared weights, spill-free | 9/9 | ~0.479 ms | 467.26 us | 2.202 GB | 81.403 M | 168 | 203.776 KB | 3.30x baseline |
| V4 | Q1/KV5 pipeline | 9/9 | 0.4675-0.4685 ms | 457.22 us | 2.202 GB | 81.318 M | 168 | 203.776 KB | robust winner |
| V6 | BLOCK_Q8, two N256 MMAs | 9/9 | 0.694-0.695 ms | 697.09 us | 1.122 GB | 47.978 M | 224 | 203.776 KB | rejected: one math WG serializes compute |
| V7 | one TMEM x64 load | 9/9 | 0.4720-0.4730 ms | not faster | 2.202 GB | comparable | 168 | 203.776 KB | rejected: x32 loads overlap better |
| V8 | explicit `ld.shared.v4.b32` | 9/9 | 0.4668-0.4681 ms | 457.18 us | 2.202 GB | 81.342 M | 168 | 203.776 KB | compiler already emitted same load count |
| V9 | remove redundant grid dependency sync | 9/9 | 0.4671-0.4683 ms | 455.62 us | 2.202 GB | comparable | 168 | 203.776 KB | small NCU lead; benchmark overlaps V4 |
| V10 | 2-CTA multicast bring-up | 9/9 | ~0.468 ms | 469.4 us | 2.202 GB class | comparable | 168 | 203.776 KB | valid, no speedup |
| V11 | multicast burst scheduling | 9/9 | no measurable win | not promoted | comparable | comparable | 168 | 203.776 KB | rejected |
| V12 | collapse heads into `q_eff` | failed | n/a | n/a | reduced | reduced | n/a | n/a | invalid: per-head ReLU prevents the transform |
| V13 | true `cta_group::2` M256, half-KV per CTA | 9/9 | 0.5715-0.5719 ms | 575.71 us | 1.228 GB | 51.275 M | 224 | 121.344 KB | valid traffic proof, slower than V4 |
| V14 | V13 plus two fixed Q-pair consumer WGs | 9/9 | 0.5189-0.5227 ms | 522.88 us | 1.228 GB | 51.249 M | 168 | 121.344 KB | 9% faster than V13, still 11% behind V4 |
| V15 | cluster Q16, four M256 passes, alternating consumer WGs | 9/9 | 0.4579-0.4591 ms | 461.98 us | 692.191 MB | 34.570 M | 168 | 155.136 KB | first two-SM traffic win near V4 latency |
| V16 | V15 with 32 KV splits per scheduler chunk | 9/9 | 0.4527-0.4536 ms | 453.92 us | 621.863 MB | 32.456 M | 168 | 155.136 KB | first two-SM benchmark winner |
| V17 | V16 with 64 KV splits per scheduler chunk | 9/9 | 0.4497-0.4509 ms | 448.48 us | 586.490 MB | 31.414 M | 168 | 155.136 KB | previous champion |
| V18 | V17 with 128 KV splits per chunk | 9/9 | 0.4497-0.4500 ms | 448.74 us | 586.490 MB | 31.350 M | 168 | 155.136 KB | rejected: no robust gain |
| V19 | V17 with six KV stages | 9/9 | 0.4503 ms average | 451.78 us | 586.490 MB | 31.379 M | 168 | 172.032 KB | rejected: deeper ring regresses |
| V20 | explicit group2 TMA completion aggregation | failed | n/a | n/a | n/a | n/a | n/a | n/a | rejected: launch failure, then barrier deadlock |
| V21 | Q32 cluster, eight M256 passes, five KV stages | 9/9 | 0.4588 ms average | 457.76 us | 326.320 MB | 23.380 M | 168 | 224.768 KB | valid traffic reduction, slower |
| V22 | V21 with four KV stages | 9/9 | 0.4571 ms average | 456.99 us | 326.320 MB | 23.392 M | 168 | 207.872 KB | faster than V21, still behind V17 |
| V23 | V17 with four KV stages | 9/9 | 0.4477 ms average | 447.52 us | 586.490 MB | 31.353 M | 168 | 138.240 KB | current champion |

V5 (Q1/KV6) is intentionally excluded because its dynamic shared-memory
request is invalid on the device and it did not pass correctness execution.

V13 and V14 are genuine two-SM implementations rather than duplicate-load
cluster experiments.  Their generated PTX contains
`tcgen05.mma.cta_group::2.kind::f8f6f4`, and the cubin disassembly contains
`UTCQMMA.2CTA`, `UTCBAR.2CTA.MULTICAST`, `UTMALDG`, and `LDTM`.  The physical
KV cache is bitwise unchanged, and the independent m4096 profile oracle records
maximum absolute error `4.76837158203125e-7`.

## Current NCU interpretation

V9 retains exact semantic evidence: maximum absolute error is
`4.76837158203125e-7`, below the formal bound, and the physical cache is
bitwise unchanged. Its full NCU report records:

- duration: `455.62 us`
- compute throughput: `61.13%`
- memory throughput: `65.64%`
- L1/TEX hit rate: `0.16%`
- L2 hit rate: `82.41%`
- registers: `168/thread`
- dynamic shared memory: `203.78 KB/block`
- achieved occupancy: `17.05%`
- local/shared spilling: zero
- active/eligible warps per scheduler: `2.73 / 0.65`

The report classifies compute and memory as balanced, but the low eligible-warp
count and repeated KV TMA traffic still expose a structural opportunity. V6
demonstrates that aggregating eight query rows can nearly halve TMA traffic and
L2 sectors, but doing so inside one CTA removes a math warp-group and raises
register pressure enough to lose performance.

## Two-SM result and next structural experiment

The V13/V14 experiments implemented a two-CTA cluster for two adjacent Q4
tiles.  Each CTA loads one disjoint M128 KV half and one cta-group::2 M256N256
instruction consumes both halves.  Relative to V4, V14 reduced TMA receiver
traffic by 44.2% and L2 sectors by 37.0%, while restoring the 168-register
budget and increasing active warps from 2.00 (V13) to 3.00.  It did not win:
tensor-pipe utilization fell from 54.64% to 46.08%, and kernel duration rose
from 457.22 us to 522.88 us.

Full-source NCU sampling locates the dominant V14 long-scoreboard hotspot at
the TMEM/full-barrier wait immediately before `LDTM`; other high-sample waits
are the peer-KV readiness barrier and the same consumer wait in the second
stage.  The next experiments therefore target readiness/issue overlap, not
additional traffic reduction:

1. remove synchronization that is proven redundant, one barrier family at a
   time, and rerun full correctness before timing;
2. test whether an additional half-KV stage hides the peer-readiness bubble
   within the 121 KB shared-memory footprint;
3. retain V4 as the production champion unless the same-run formal medians and
   full NCU report both improve without cache mutation or tolerance changes.

This targets the V6 traffic reduction without accepting V6's single-math-WG
serialization. A duplicate-load cluster will only be used as a scheduling and
correctness bring-up point; it is not a performance candidate.

## Q16 cluster and scheduler-chunk result

V15 extends the genuine two-SM path to one Q8 tile per CTA, or Q16 per cluster,
and issues four `cta_group::2` M256N256 passes for each KV tile.  Two consumer
warp groups alternate the four Q pairs over two TMEM stages.  Because the SM100
TMA 2D box cannot legally cover Q8 in one transfer, Q is loaded as two Q4 TMA
transactions; this does not change the per-head ReLU, FP32 accumulation and
reduction, scale placement, causal mask, or output order.

Increasing only the scheduler chunk from 16 to 32 and then 64 KV splits reduced
loop/scheduling overhead and repeated cache traffic.  The clean three-seed
formal averages for V17 are `0.119235`, `0.230809`, and `0.450416 ms` for
`m=1024/2048/4096`, respectively.  Relative to V16 this is a further
`1.87% / 0.63% / 0.61%` reduction.  V17's full m4096 NCU report records:

- duration: `448.480 us`
- DRAM read/write: `46.979 / 203.773 MB`
- TMA receiver traffic: `586.490 MB`
- L1 sectors and hit rate: `8.220 M`, `0.120%`
- L2 sectors and hit rate: `31.414 M`, `53.396%`
- tensor-pipe activity: `54.510%`
- SM throughput: `60.971%`
- active warps per scheduler: `2.996`
- registers and dynamic shared memory: `168/thread`, `155.136 KB/block`

The independent oracle maximum absolute error remains
`4.76837158203125e-7`, and the physical KV cache remains bitwise unchanged.
Generated PTX contains 16 `tcgen05.mma.cta_group::2` instructions; generated
SASS contains 16 `UTCQMMA.2CTA`, four `UTCBAR.2CTA.MULTICAST`, seven `UTMALDG`,
and 16 `LDTM` instructions.  Full-source sampling still places the largest
long-scoreboard samples on the polling branches immediately following
`SYNCS.PHASECHK.TRANS64.TRYWAIT`, before the TMEM loads.  The next controlled
experiment therefore increases the chunk once more, while retaining V17 if
the longer readiness interval stops paying for the saved scheduler traffic.

## V18-V22 report-driven follow-up

V18 increased the scheduler chunk from 64 to 128.  Its clean formal averages
were `0.119393 / 0.230912 / 0.449841 ms` at `m=1024/2048/4096`.  The largest
case was only `0.128%` faster than V17 in the benchmark while its full NCU
duration was `448.736 us`, slightly slower than V17.  It therefore does not
replace the more robust chunk-64 point.

V19 increased the KV ring from five to six stages.  All nine cases passed, but
the averages became `0.119953 / 0.231040 / 0.450266 ms`, and full NCU rose to
`451.776 us` despite essentially unchanged traffic.  The additional
`16.896 KB` of SMEM did not hide a new latency component, so the deeper ring
was rejected.

V20 attempted to replace the rank-1 local KV wait plus peer arrival with
explicit `cta_group::2` TMA transaction aggregation.  Initial barrier arrival
count 1 caused an unspecified launch failure.  Matching the official group2
initial count of 2 removed that failure but deadlocked in the first strict
case.  No performance claim or NCU report is made for a kernel that cannot
complete.  Both failure diagnostics and the source are preserved.

V21 doubled the reuse window from Q16 to Q32 while retaining the exact
M256N256K32 group2 MMA and per-head ReLU arithmetic.  It passed 9/9, with
maximum independent profile-oracle error `4.76837158203125e-7` and bitwise
unchanged cache.  Formal averages were
`0.125077 / 0.236281 / 0.458832 ms`.  Full NCU confirms the intended traffic
effect: TMA receiver traffic fell from `586.490` to `326.320 MB` and L2 sectors
from `31.414` to `23.380 M`.  However, duration increased from `448.480` to
`457.760 us`; tensor activity fell from `54.510%` to `52.931%`, SM throughput
from `60.971%` to `59.216%`, and dynamic SMEM grew from `155.136` to
`224.768 KB`.  Source sampling exposes four dominant TMEM-ready polling sites
instead of V17's two, showing that the eight-pass Q-pair dependency chain, not
KV traffic, becomes the limiting path.

V22 reduced the Q32 KV ring from five stages to four, lowering dynamic SMEM to
`207.872 KB`.  It remained 9/9 and improved the clean formal averages to
`0.124425 / 0.235284 / 0.457117 ms`; NCU improved to `456.992 us`.  TMA traffic
remained `326.320 MB`, L2 sectors `23.392 M`, tensor activity `52.834%`, and SM
throughput `59.109%`.  This confirms that five stages were excessive for Q32,
but also that reducing L2 traffic alone cannot overcome the longer TMEM and
epilogue critical path.  V17 remains the production champion.

## V23 Q16 four-stage KV ring

V23 applies the V22 ring-depth finding to the lower-latency Q16 path: the only
device scheduling change from V17 is `kNumKVStages=5` to `4`.  It passed all
nine formal cases at the unchanged `rtol=1e-5`, `atol=1e-6`; no other compute
process was observed.  Clean candidate median averages were
`0.118771 / 0.229259 / 0.447697 ms` for `m=1024/2048/4096`, improving over V17
by approximately `0.39% / 0.67% / 0.60%`.

The full m4096 NCU report measures `447.520 us`, down from V17's `448.480 us`.
Dynamic SMEM falls from `155.136` to `138.240 KB/block`, while registers remain
`168/thread`.  TMA receiver bytes and L1 sectors are exactly unchanged at
`586.490 MB` and `8.220 M`; L2 sectors decrease slightly from `31.414` to
`31.353 M`.  Tensor-pipe activity improves from `54.510%` to `54.732%`, and SM
throughput from `60.971%` to `61.219%`.  PC sampling records 28,541 samples:
10,053 long-scoreboard, 5,145 wait, 1,241 short-scoreboard, 1,722 sleeping, and
1,616 barrier.  The shorter ring does not remove the dominant TMEM-ready wait,
but it reduces short-scoreboard/sleep overhead enough to improve the critical
path.  NSYS independently records a `444.639 us` kernel versus V17's
`446.431 us`.

The independent profile oracle remains at maximum absolute error
`4.76837158203125e-7`, and the physical cache is bitwise unchanged.  Generated
PTX contains 16 `tcgen05.mma.cta_group::2` instructions and SASS contains 16
`UTCQMMA.2CTA`.  V23 therefore replaces V17 as the production champion.

## Evidence locations

Full reports are intentionally kept outside Git history under:

`results/deepseek_v4_chunked_mega_mqa_logits/<version>_single_request_20260731/`

The V9 report is:

`results/deepseek_v4_chunked_mega_mqa_logits/blockq4_nogriddep_q1kv5_v9_single_request_20260731/ncu/m4096_blockq4_nogriddep_q1kv5_v9_set_full.ncu-rep`

The genuine two-SM reports are:

- `results/deepseek_v4_chunked_mega_mqa_logits/cluster2_q8_2smkv_dualq4_q1kv5_v13_single_request_20260731/ncu/m4096_cluster2_q8_2smkv_dualq4_q1kv5_v13_set_full.ncu-rep`
- `results/deepseek_v4_chunked_mega_mqa_logits/cluster2_q8_2smkv_2wg_q1kv5_v14_single_request_20260731/ncu/m4096_cluster2_q8_2smkv_2wg_q1kv5_v14_set_full.ncu-rep`

The current Q16 reports are:

- `results/deepseek_v4_chunked_mega_mqa_logits/cluster2_q16_2smkv_2wg_q1kv5_v15_single_request_20260731/`
- `results/deepseek_v4_chunked_mega_mqa_logits/cluster2_q16_2smkv_2wg_q1kv5_chunk32_v16_single_request_20260731/`
- `results/deepseek_v4_chunked_mega_mqa_logits/cluster2_q16_2smkv_2wg_q1kv5_chunk64_v17_single_request_20260731/`
- `results/deepseek_v4_chunked_mega_mqa_logits/cluster2_q16_2smkv_2wg_q1kv5_chunk128_v18_single_request_20260731/`
- `results/deepseek_v4_chunked_mega_mqa_logits/cluster2_q16_2smkv_2wg_q1kv6_chunk64_v19_single_request_20260731/`
- `results/deepseek_v4_chunked_mega_mqa_logits/cluster2_q32_2smkv_2wg_q1kv5_chunk64_v21_single_request_20260731/`
- `results/deepseek_v4_chunked_mega_mqa_logits/cluster2_q32_2smkv_2wg_q1kv4_chunk64_v22_single_request_20260731/`
- `results/deepseek_v4_chunked_mega_mqa_logits/cluster2_q16_2smkv_2wg_q1kv4_chunk64_v23_single_request_20260731/`

Mirrored local archives are under `D:\work\agent4kernel\mega_results\v18_*`
through `v23_*`; V20 failures are under `mega_results\v20_failure`.
