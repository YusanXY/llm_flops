# FlashInfer 1x1 BSR sparse prefill candidate

This candidate flattens logical query heads into independent BSR rows (`R=1`)
and maps each selected KV token to a 1x1 column block. Each original sparse
index row is repeated 128 times at plan time so its 128 heads keep the same
selection. The original KV tensor is passed as both key and value, avoiding KV
duplication or a gathered KV tensor.

`BlockSparseAttentionWrapper.plan`, its 128 MiB workspace, the flattened
indices, and the output buffer are cached per input indices tensor. Repeated
benchmark calls execute only `BlockSparseAttentionWrapper.run`.

The backend is pinned to `fa2`, while the `R=1`, one-Q-head layout selects the
wrapper's CUDA-core path. FlashInfer 0.6.12 tensor-core FA2/FA3 configurations
are invalid for this SM100 `head_dim=512` contract.
