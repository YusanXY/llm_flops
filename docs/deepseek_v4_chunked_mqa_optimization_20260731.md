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

V5 (Q1/KV6) is intentionally excluded because its dynamic shared-memory
request is invalid on the device and it did not pass correctness execution.

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

## Next structural experiment

The next experiment is a two-CTA cluster for two adjacent Q4 tiles:

1. preserve two math warp-groups per CTA and the spill-free V4 epilogue;
2. pair adjacent Q4 CTAs that consume the same physical KV span;
3. have one cluster rank issue each KV TMA transfer with multicast to both
   CTAs' shared-memory stages;
4. retain per-CTA Q loads, masks, FP32 accumulation, scale placement, and output
   ordering;
5. launch with an explicit cluster dimension of two and prove the multicast in
   PTX/SASS and NCU before considering promotion.

This targets the V6 traffic reduction without accepting V6's single-math-WG
serialization. A duplicate-load cluster will only be used as a scheduling and
correctness bring-up point; it is not a performance candidate.

## Evidence locations

Full reports are intentionally kept outside Git history under:

`results/deepseek_v4_chunked_mega_mqa_logits/<version>_single_request_20260731/`

The V9 report is:

`results/deepseek_v4_chunked_mega_mqa_logits/blockq4_nogriddep_q1kv5_v9_single_request_20260731/ncu/m4096_blockq4_nogriddep_q1kv5_v9_set_full.ncu-rep`
