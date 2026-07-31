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

## Evidence locations

Full reports are intentionally kept outside Git history under:

`results/deepseek_v4_chunked_mega_mqa_logits/<version>_single_request_20260731/`

The V9 report is:

`results/deepseek_v4_chunked_mega_mqa_logits/blockq4_nogriddep_q1kv5_v9_single_request_20260731/ncu/m4096_blockq4_nogriddep_q1kv5_v9_set_full.ncu-rep`

The genuine two-SM reports are:

- `results/deepseek_v4_chunked_mega_mqa_logits/cluster2_q8_2smkv_dualq4_q1kv5_v13_single_request_20260731/ncu/m4096_cluster2_q8_2smkv_dualq4_q1kv5_v13_set_full.ncu-rep`
- `results/deepseek_v4_chunked_mega_mqa_logits/cluster2_q8_2smkv_2wg_q1kv5_v14_single_request_20260731/ncu/m4096_cluster2_q8_2smkv_2wg_q1kv5_v14_set_full.ncu-rep`
