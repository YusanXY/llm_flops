# DeepSeek V4 FP8 paged MQA logits

The reference is the legacy `DeepGEMM fp8_paged_mqa_logits` path wrapped by
SGLang's split helper. The contract models the C4 cache explicitly: raw
context is rounded up by four, the compressed sequence is paged in blocks of
64, and tail-page tokens are excluded by `context_lens`.

The cache, block table, and schedule are allocated once per isolated input
clone. Cache mutation is forbidden and small cache views are included in
`observed_state`. `workspace_bytes` is deliberately unknown (`None`) because
the backend does not expose a stable workspace size. The `auto` timer attempts
CUDA Graph capture and records an event-timer fallback when addresses or the
backend are not capture-safe.

Small cases use non-zero, exactly representable FP8 query/cache values, a
non-trivial head weight, reversed page order for one request, and poisoned tail
padding. Their logits are checked against an independent dense PyTorch oracle.
The 65,536-token case keeps only 256 deterministic oracle samples; the oracle
is read-only state that is never passed to candidate code.

Prefill projection cases model exactly one request with `m=1024/2048/4096`
causal query positions: Q is `[1,m,64,128]`, lengths are `[1,m]`, and the
physical block table is `[1,256]`.  Decode cases retain true batched-request
semantics.  No prefill CaseSpec expands the `m` token positions into `m`
pseudo requests or allocates `m` disjoint copies of the C4 cache.
