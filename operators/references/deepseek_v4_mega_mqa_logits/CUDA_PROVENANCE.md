# CUDA provenance for the sequential mega-MQA baseline

> Contract note: the saved first-publication Nsight report described below
> belongs to contract v1 (`aligned_decode_batch`), which incorrectly modeled
> `M` independent requests.  Contract v2 is one causal request with `M`
> contiguous query tokens and shared physical pages.  The v1 report is
> historical evidence only and must not be used to rank v2 candidates.

This ledger traces every semantic member of `implementation.py` to the
lowest CUDA implementation that is actually reachable in the control
baseline.  It distinguishes the production SGLang kernel that defined the
contract from the deliberately unfused PyTorch implementation used by the
reference.  Nsight Compute kernel names are the final runtime authority; the
saved `--set full` report and its exported kernel list complement this file.

## Runtime pinned for the first control publication

- llm_flops commit: `417a4c019dabf2737b1b5cda859a1bb8e1d2c57d`
- PyTorch: `2.11.0+cu130`, commit
  `70d99e998b4955e0049d13a98d77ae1b14db1f45`
- DeepGEMM: `0.1.4`
- GPU: NVIDIA B200 (SM100)
- Nsight Compute: `/opt/nvidia/nsight-compute/2026.1.1/ncu`

## Member-by-member implementation trace

| # | Semantic member | Control-baseline CUDA path | Lowest source/library implementation | Production SGLang contract source |
|---:|---|---|---|---|
| 1 | `X @ W_gate^T`, BF16 inputs and FP32 output | `torch.mm(..., out_dtype=torch.float32)` -> ATen CUDA BLAS dispatch -> cuBLAS/cuBLASLt | PyTorch `aten/src/ATen/native/cuda/Blas.cpp`, `aten/src/ATen/cuda/CUDABlas.cpp`, `aten/src/ATen/cuda/CUDABlasLt.cpp`; the terminal GEMM is an NVIDIA library kernel selected at runtime | `python/sglang/srt/models/deepseek_v4.py` constructs the Indexer compressor `wkv_gate`; the C4 compressor consumes its 512-value output |
| 2 | C4 ring-state write | `Tensor.copy_` | PyTorch `aten/src/ATen/native/cuda/Copy.cu`, with CUDA elementwise/copy templates in `aten/src/ATen/native/cuda/CUDALoops.cuh` and `aten/src/ATen/cuda/detail/KernelUtils.h` | `python/sglang/jit_kernel/csrc/deepseek_v4/c4.cuh` |
| 3 | Select overlap/regular K and score fields | views plus `torch.cat` | PyTorch `aten/src/ATen/native/cuda/Shape.cu` (`CatArrayBatchedCopy`/cat copy path) | `python/sglang/jit_kernel/csrc/deepseek_v4/c4.cuh` |
| 4 | Per-dimension 8-way softmax | pointwise score add, then `torch.softmax(dim=1)` | PyTorch `aten/src/ATen/native/cuda/SoftMax.cu`, `aten/src/ATen/native/cuda/PersistentSoftmax.cuh`, and CUDA elementwise loop templates | `python/sglang/jit_kernel/csrc/deepseek_v4/c4.cuh` |
| 5 | Weighted 8-token reduction | pointwise multiply followed by `torch.sum(dim=1)` | PyTorch `aten/src/ATen/native/cuda/ReduceSumProdKernel.cu` and `aten/src/ATen/native/cuda/Reduce.cuh` | `python/sglang/jit_kernel/csrc/deepseek_v4/c4.cuh` |
| 6 | RMSNorm over 128 values | `square`, `mean`, `rsqrt`, multiply | PyTorch CUDA pointwise loops plus `ReduceSumProdKernel.cu`/`Reduce.cuh` | `python/sglang/jit_kernel/csrc/deepseek_v4/fused_norm_rope.cuh` |
| 7 | RoPE on trailing 64 values | views, pointwise multiply/add/subtract, `copy_` | PyTorch CUDA elementwise loops and `Copy.cu` | `python/sglang/jit_kernel/csrc/deepseek_v4/fused_norm_rope.cuh` |
| 8 | Normalized 128-point FWHT | seven explicit butterfly levels using pointwise add/subtract and `cat` | PyTorch CUDA elementwise loops plus `Shape.cu` cat copies | `python/sglang/jit_kernel/csrc/fast-hadamard-transform/hadamard_jit.cuh` |
| 9 | Per-token absmax and FP32 K scale | `abs`, `amax`, clamp, divide | PyTorch CUDA pointwise loops; reduction in `aten/src/ATen/native/cuda/ReduceMaxValuesKernel.cu`/`Reduce.cuh` | quantize/store contract in `python/sglang/jit_kernel/csrc/deepseek_v4/store.cuh` |
| 10 | E4M3 quantization | clamp followed by cast to `torch.float8_e4m3fn` | PyTorch CUDA pointwise/copy conversion templates (`CUDALoops.cuh`, `Copy.cu`, CUDA numeric conversion helpers) | `python/sglang/jit_kernel/csrc/deepseek_v4/store.cuh` |
| 11 | Physical 8448-byte-page scatter | advanced-index assignment for FP8 codes and FP32 scales | PyTorch `aten/src/ATen/native/cuda/Indexing.cu` and `aten/src/ATen/native/cuda/ScatterGatherKernel.cu` | `python/sglang/jit_kernel/csrc/deepseek_v4/store.cuh` |
| 12 | FP8 paged MQA logits | `deep_gemm.fp8_paged_mqa_logits` -> compiled `_C` binding -> SM100 JIT kernel | `deep_gemm/include/deep_gemm/impls/sm100_mqa_logits.cuh`; paged scheduler in `deep_gemm/include/deep_gemm/scheduler/sm100_paged_mqa_logits.cuh`; cache layout in `deep_gemm/include/deep_gemm/layout/mqa_logits.cuh`; PTX wrappers in `deep_gemm/include/deep_gemm/ptx/tcgen05.cuh` and TMA headers | This DeepGEMM kernel is the production consumer |

## SM100 MQA facts verified in the installed source

- Python entry:
  `deep_gemm/__init__.py:206-207`.
- Metadata entry:
  `deep_gemm/__init__.py:223-224`.
- TCGen05 header inclusion:
  `sm100_mqa_logits.cuh:14`.
- TCGen05 synchronization/commit sites:
  `sm100_mqa_logits.cuh:300,337,423,447`.
- BF16 reduction ReLU:
  `sm100_mqa_logits.cuh:456`.
- FP32 reduction ReLU uses `(x + abs(x)) / 2`:
  `sm100_mqa_logits.cuh:474`.
- The implementation comments explicitly require preserving KV shared memory
  until UMMA has committed its accumulators to TMEM.

## Important interpretation

The control candidate is intentionally **not** a fused CUDA implementation.
Stages 2-11 expand into many ATen CUDA kernels and materialize all
intermediates.  This makes it a valid semantic/performance control and lets
the first full NCU report quantify launch, reduction, copy, indexing, and
memory-traffic costs separately.  A later mega-kernel candidate may replace
those paths, but it must preserve:

1. the C4 overlap semantics,
2. FP32 RMS/scale/reduction behavior,
3. exact E4M3 K codes and the 8448-byte page ABI,
4. per-head ReLU before weighted head reduction, and
5. context masking.

## Runtime closure from the first full NCU report

The m=4096 `--set full` capture resolved 61 launches.  The observed sequence
closed the static trace above:

- ID 0: `nvjet_sm100_tss_128x128_64x9_2x2_2cta_h_bz_TNT`, the NVIDIA
  library GEMM selected by `torch.mm`.
- IDs 1-59: ATen `elementwise_kernel`, `CatArrayBatchedCopy`,
  `cunn_SpatialSoftMaxForward`, `reduce_kernel`,
  `float8_copy_kernel_cuda`, and `index_elementwise_kernel` variants.
- ID 60:
  `deep_gemm::sm100_paged_mqa_logits<...>`.

The SASS source page exported from the report contains `UTMALDG.2D`,
`UTMALDG.3D`, `UTCQMMA`, `UTCBAR`, and TMEM operands.  This is direct runtime
evidence that the final consumer uses SM100 TMA, TCGen05/UMMA, and TMEM; it is
not merely inferred from a header or a kernel name.
