# DeepSeek V4 TopK transform

The reference is the legacy optimized SGLang `plan_topk_v2` plus
`topk_transform_512_v2` path. Selection is by descending score with raw-index
ascending cutoff ties, then each raw index is transformed through the page
table. The public contract treats each batch row as an unordered set and does
not expose scores. Page-table ranges are disjoint across batches so a batch
mix-up cannot pass correctness accidentally.

The SGLang private JIT cache is materialized while the worker imports the
reference entrypoint. Ninja/PTXAS therefore appears in `import_ms`; metadata,
page tables and output allocation are prepared before timing. Only the public
optimized kernel is captured and sampled in steady state.
The control candidate is a byte-identical copy of this implementation,
so its expected performance ratio is approximately 1x.

Prefill projection CaseSpecs describe one request with `m` causal query
tokens.  The SGLang row-wise transform ABI still receives `m` score rows and
an internal contiguous expansion of the single request's page table, but all
expanded rows contain the same physical page IDs.  This is an ABI adapter,
not `m` requests; decode CaseSpecs retain true batched-request semantics.
