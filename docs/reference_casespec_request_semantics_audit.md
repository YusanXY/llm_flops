# Reference CaseSpec request-semantics audit

Date: 2026-07-31

This audit covers every `operators/references/*/spec.py` in the repository.
The invariant used for sequence operators is:

- request count and query-token count are distinct dimensions;
- a one-request prefill with `m` tokens must not allocate `m` independent KV
  caches or `m` disjoint physical page tables;
- per-token causal metadata (lengths and positions) must vary along the token
  dimension;
- an implementation-only row expansion is allowed only when all expanded
  rows alias the same request-owned physical pages and the CaseSpec records
  that it is an ABI adapter.

| Reference | Request/token interpretation | Verdict |
|---|---|---|
| `deepseek_v4_chunked_mega_mqa_logits` | One request, Q `[1,m,64,128]`, lengths `[1,m]`, one page row | Fixed before this audit; correct |
| `deepseek_v4_dense_swa_attention` | Decode `batch` is a true request batch | Correct |
| `deepseek_v4_fp8_gemm_nt` | `m` is the GEMM row/token dimension; no request-owned state | Correct / request-insensitive |
| `deepseek_v4_fp8_paged_mqa_logits` | Prefill is one request with `m` causal tokens; decode is a true request batch | Fixed prefill Q/length/page-table semantics; decode unchanged |
| `deepseek_v4_indexer_fp8_quant` | Stateless row kernel; prefill rows are contiguous tokens of one request | Fixed prefill metadata and causal RoPE positions; decode unchanged |
| `deepseek_v4_mega_mqa_logits` | One request covering the producer, physical cache and MQA consumer | Fixed Q/length/page-table dimensions |
| `deepseek_v4_sparse_decode_attention` | Decode `batch` is a true request batch with per-request cache regions | Correct |
| `deepseek_v4_sparse_prefill_attention` | Flat query-token rows consume one shared KV sequence | Correct |
| `deepseek_v4_topk_transform` | Prefill has `m` score rows for one request; decode is a true request batch | Fixed causal lengths and shared physical-page aliases; row expansion documented as ABI-only |
| `deepseek_v4_trtllm_fp8_mxfp8_moe` | `m` is a routed token-row dimension | Correct / request-insensitive |
| `example_cpu_add` | Element count only | Correct / request-insensitive |
| `glm5_deepep_dispatch` | Distributed routed-token contract; local input construction is explicitly unsupported | Correct / request-insensitive |
| `glm5_dense_prefill_attention` | Query `[1,q,...]`, one block table and one KV length | Correct |
| `glm5_dsa_index_score` | Prefill uses `m` query rows over one dense cache; decode uses a true paged request batch | Correct |
| `glm5_dsa_indexer` | `gemm_m` is a projection row/token dimension | Correct / request-insensitive |
| `glm5_dsa_projection` | `m` is a GEMM/BMM row dimension; explicit BMM batch is separate | Correct / request-insensitive |
| `glm5_dsa_sparse_attention` | Flat query-token rows consume one shared KV sequence | Correct |
| `glm5_dsa_unified_sparse_attention` | Flat query-token rows consume one shared KV sequence | Correct |
| `glm5_moe_grouped_gemm` | Physical rows are expert-routed tokens | Correct / request-insensitive |
| `glm5_moe_masked_grouped_gemm` | Masked expert rows are routed tokens | Correct / request-insensitive |

The audit intentionally does not relabel legitimate decode batches or matrix
row dimensions as request errors.  It also does not change tolerances,
benchmark scripts, operator mathematics, or backend selection.

## Validation

- Registry validation: 20 references and 39 discovered candidates.
- Full correctness after the request-semantics fixes:
  - `deepseek_v4_fp8_paged_mqa_logits`: 9/9 passed.
  - `deepseek_v4_topk_transform`: 18/18 passed.
  - `deepseek_v4_indexer_fp8_quant`: 15/15 passed.
  - `deepseek_v4_mega_mqa_logits`: 9/9 passed.
- Infrastructure failures: 0 in all four runs.
