# DeepSeek V4 full-KV-reuse / hybrid-recompute MQA boundary

This reference is the control for the smallest boundary that both:

1. preserves the original DeepSeek V4 C4 Indexer and paged-MQA semantics; and
2. exposes an upstream representation smaller than the MQA kernel's aggregate
   repeated K traffic.

It stops at `mqa_logits`.  The timed baseline is **complete physical KV
reuse**.  It never regenerates or rewrites the Indexer cache.

## Formal cases

Each case is one causal prefill request ending at raw context 65536.  The
ABI encodes that literally: Q is `[1,m,64,128]`, causal lengths are `[1,m]`,
and the page table is `[1,256]`.  There is no `m x 1` pseudo-request batch.

| `m` | raw query prefix | physical C4 rows | query rows |
|---:|---:|---:|---:|
| 1024 | 64512 | 16384 | 1024 |
| 2048 | 63488 | 16384 | 2048 |
| 4096 | 61440 | 16384 | 4096 |

Query row `i` uses

`L_i = floor((raw_prefix + i + 1) / 4)`

valid C4 records.  The one request has one physical page table.  The output is FP32
`[m, 16384]`; because SGLang uses `clean_logits=False`, only
`output[i, :L_i]` is defined.  The formal gate remains `rtol=1e-5`,
`atol=1e-6`.

## Dual, exactly equivalent K representations

The input bundle contains both:

- `kv_fused`: all 16384 C4 K records in SGLang's physical page ABI;
- `history_hidden[65536,7168]` BF16 plus `wkv_gate[512,7168]` BF16,
  APE, RMSNorm weight, and full compressed-position RoPE rows.

The CaseSpec constructs `kv_fused` from those producer inputs before timing.
The candidate-side reconstruction helper and an independent spec oracle must
recover the same E4M3 code and FP32 scale for every C4 row.

For projected record

`z_t = hidden_t @ wkv_gate^T`

split

`z_t = [K_overlap, K_regular, s_overlap, s_regular]`.

For C4 group `g>0`, the eight softmax entries use overlap fields from raw
tokens `4g-4..4g-1` and regular fields from `4g..4g+3`.  At `g=0`, the four
missing overlap values are zero and their scores are `-inf`, exactly matching
`c4_v2.cuh`.  The pooled K then executes:

`C4 -> RMSNorm -> trailing-64 RoPE(position=4g) -> normalized FWHT128`

followed by one E4M3 group and one FP32 scale per row.

The 8448-byte physical page stores:

- `64*128 = 8192` E4M3 K bytes;
- `64*4 = 256` FP32 scale bytes.

## Timed reference equation

The reference ignores the producer representation and reads only the complete
physical cache:

`d_{i,p,h} = sum_r float(Q[i,h,r]) * float(K[p,r])`

`logit[i,p] = K_scale[p] * sum_h weight[i,h] * ReLU(d_{i,p,h})`

for `p < L_i`.  The ReLU is per-head and occurs before weighted head
reduction.

`Q` and `weight` deliberately remain the exact post-fused-Q operands of
SGLang's Indexer.  This boundary targets the high-volume K path; it does not
add unrelated Q-projection operators.

## Candidate freedom

A candidate may choose a raw-token split `R`:

- reuse prefix C4 rows from `kv_fused`;
- regenerate a suffix from `history_hidden` and producer parameters;
- consume regenerated exact FP8 K+scale on chip while preserving causal
  visibility and the physical cache bytes.

This supports continuous traffic/compute tuning from full reuse (`R=0`) to
full historical recomputation.  It does not permit changing quantization,
ReLU order, C4 windows, RoPE positions, cache ABI, or accuracy tolerance.

At `m=4096`, `history_hidden` is about 0.94 GB, the physical cache is about
2.16 MB, while the causal MQA K stream contains roughly 8.5 GB of repeated
K+scale reads.  The latter is aggregate execution traffic, not unique input
storage.

## Locked semantic sources

- `sglang/srt/models/deepseek_v4.py`
- `sglang/srt/layers/attention/deepseek_v4_backend.py`
- `sglang/srt/layers/attention/dsv4/metadata.py`
- `sglang/srt/layers/attention/dsv4/indexer.py`
- `sglang/srt/layers/attention/dsv4/compressor_v2.py`
- `sglang/jit_kernel/csrc/deepseek_v4/c4_v2.cuh`
- `sglang/jit_kernel/csrc/deepseek_v4/fused_norm_rope_v2.cuh`

`sglang_semantic_audit.py` checks both the special sequence-start C4 group and
an ordinary suffix-with-carry group against the installed SGLang kernels.
