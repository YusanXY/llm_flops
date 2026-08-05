# DeepSeek-V3.2 DSA sparse attention

This reference reproduces the official FlashInfer baseline for the MLSys 2026
FlashInfer AI Kernel Generation Contest definition
`dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64`.

The timed implementation is the contest's `flashinfer_wrapper_5af199`: it
concatenates `q_nope/q_pe` and `ckv_cache/kpe_cache`, derives valid lengths from
the `-1`-padded sparse token indices, and calls
`flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla` with
`sparse_mla_top_k=2048`. The 128 MiB workspace is cached per CUDA device.

Contract:

- query heads: 16 (DeepSeek's 128 heads under TP=8)
- compressed-KV dimension: 512
- positional-key dimension: 64
- page size: 64
- sparse capacity: 2048 physical token indices per query token
- official workload geometry: 8462 pages and 1, 2, 6, 7, or 8 query tokens
- output: BF16 `[num_tokens, 16, 512]`

The operator is intentionally separate from `deepseek_v4_sparse_prefill_attention`.
That operator uses 128 heads, one 512-wide token-level KV tensor and an SGL
Kernel FlashMLA reference, so it is not shape- or backend-compatible with this
contest baseline.

Provenance:

- dataset: https://huggingface.co/datasets/flashinfer-ai/mlsys26-contest
- baseline commit: https://huggingface.co/datasets/flashinfer-ai/mlsys26-contest/commit/5e832ce88dd1013032b22a0e942979d7d2769cc3
- solution: `flashinfer_wrapper_5af199`
