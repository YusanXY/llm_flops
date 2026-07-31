# DeepSeek V4 mega MQA logits baseline

This reference is the minimum closed operator set that starts at the complete
C4 Indexer KV producer, crosses the physical cache entrance, consumes that
cache with paged MQA, and stops at FP32 `mqa_logits`.

It is intentionally a sequential baseline.  `implementation.py` imports
PyTorch and DeepGEMM directly and does not call another llm_flops reference or
an SGLang high-level operator.  The C4 compressor, normalization, RoPE,
Hadamard, quantization, physical scatter, and DeepGEMM call are separate
functions so profiling can attribute every unfused boundary.

## Closed operator boundary

Included, in execution order:

1. BF16 `wkv_gate`: `X[M,7168] @ W[512,7168]^T -> G[M,512]` with FP32 output.
2. Single-request C4 ring-state update by all `M` contiguous raw records.
3. `M/4` aligned eight-token overlap compressions.  The oldest four tokens use
   overlap K/score fields; the newest four use regular K/score fields.
4. FP32 RMSNorm over 128 dimensions.
5. RoPE on the trailing 64 dimensions at positions
   `raw_context-M, raw_context-M+4, ..., raw_context-4`.
6. Normalized 128-point FWHT.
7. Per-token E4M3 quantization with
   `scale=max(1e-4,max(abs(K)))/448`.
8. Scatter into the production 8448-byte page ABI: 8192 contiguous FP8 K
   bytes, then 256 bytes containing 64 FP32 scales.
9. DeepGEMM FP8 paged MQA, including per-head ReLU, weighted head reduction,
   FP32 accumulation/output, and context masking.

Excluded: Q projection/quantization, head-weight projection, TopK transform,
sparse attention, the ordinary attention KV producer, and C128 compression.
The external `q_fp8` and `weights` inputs are already in the exact ABI consumed
by MQA.

## Mathematical contract

There is exactly one request.  Let `P=65536-M` be its existing raw prefix,
`j=0..M-1` index its contiguous query/extend tokens, `r=0..M/4-1` index newly
completed C4 records, `d` index a dimension, and `h` index a head:

```text
G_j = X_j W_gate^T
records = concat(previous_request_state[-4:], G_0, ..., G_{M-1})
state = records[-8:]

Kbar_rd =
  sum_{i=0..7} softmax_i(score_bid + APE_id) * value_bid

Knorm_r = RMSNorm(Kbar_r)
Krope_r = RoPE_position=P+4r(Knorm_r)
Krot_r  = H_128(Krope_r) / sqrt(128)

sK_r = max(1e-4, max_d |Krot_rd|) / 448
K8_r = E4M3(clamp(Krot_r / sK_r, -448, 448))
```

The `M/4` newly produced `(K8_r,sK_r)` records fill compressed cache positions
`P/4 ... 16383` of the request's single 256-page physical cache.  Query `j`
has causal compressed length `L_j=floor((P+j+1)/4)`.  MQA computes, only for
the valid ragged region `t<L_j`,

```text
logits[j,t] =
  sum_h weights[j,h] *
        ReLU(sum_d float(q8[j,h,d]) * float(K8[t,d])) *
        sK[t]
```

The ReLU is part of the DeepGEMM SM100 reduction and is not optional.

## Formal CaseSpecs

Exactly three formal cases are declared:

| case | requests | final raw context | query tokens | new C4 K |
|---|---:|---:|---:|---:|
| `formal__mega_mqa_logits__m1024__ctx65536` | 1 | 65536 | 1024 | 256 |
| `formal__mega_mqa_logits__m2048__ctx65536` | 1 | 65536 | 2048 | 512 |
| `formal__mega_mqa_logits__m4096__ctx65536` | 1 | 65536 | 4096 | 1024 |

The ABI carries one logical page-table row for the one request.  Its `m`
query positions occupy Q `[1,m,64,128]` and causal lengths `[1,m]`; there is
no `m x 1` pseudo-request expansion.  All positions therefore address the
same 256 physical pages and differ only in their causal `context_lens`.
Correctness uses `rtol=1e-5`, `atol=1e-6`; physical FP8 K codes and page-table
state must be exact, while physical FP32 scales use the same strict floating
gate.  Logits outside each query's ragged valid length are intentionally
unobserved, matching SGLang's `clean_logits=False` consumer contract.  The
timer is CUDA-event based and all sequential intermediates are inside the
timed callable.  `mode=single_request_causal_prefill` is explicit in the
CaseSpec.
