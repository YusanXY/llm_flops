"""Contract, deterministic cases and per-batch comparator for TopK transform."""

import importlib

from benchmark_engine.correctness import InputBundle
from benchmark_engine.correctness.models import ComparisonResult, OutputBundle, OutputLeaf
from benchmark_engine.models import CaseSpec


LEGACY_TOPK = {
    "adapter_name": "C4 TopK Transform",
    "backend": "SGLang JIT topk_transform_512_v2",
    "instances": 30,
    "prefill_m": (1024, 2048, 4096),
    "decode_m": (16, 32),
    "context": 65536,
    "topk": 1024,
    "page_size": 64,
}


def _projection_cases():
    return tuple(CaseSpec(
        f"{phase}__c4_topk_transform__m{m}__ctx65536__fp8_mxfp8",
        {"batch": 1 if phase == "prefill" else m,
         "query_tokens": m if phase == "prefill" else 1,
         "request_count": 1 if phase == "prefill" else m,
         "length": 16384, "seq_len": 16384, "topk": 1024, "page_size": 64,
         "pattern": "random", "phase": phase, "quant_profile": "fp8_mxfp8",
         "raw_context": 65536, "model_input": m, "projection_adapter_id": "c4_topk_transform"},
        227, frozenset({"model_projection", f"deepseek_v4_{phase}", f"phase_{phase}",
                        "quant_profile_fp8_mxfp8", f"m_{m}", "context_65536", "performance_only"}), 1800)
        for phase, inputs in (("prefill", (1024, 2048, 4096)), ("decode", (16, 32))) for m in inputs)


class BatchedUnorderedTopKComparator:
    """Compare each batch row as a set; never flatten across batch boundaries."""

    def __init__(self, *, sampled=False):
        self.sampled = sampled

    def compare(self, reference, candidate, **_):
        refs, cands = reference.by_path(), candidate.by_path()
        path = "output"
        if set(refs) != {path} or set(cands) != {path}:
            return ComparisonResult(False, "batched_unordered_topk", {"error": "output structure mismatch"}, failed_path=path)
        left, right = refs[path], cands[path]
        if left.shape != right.shape or len(left.shape) != 2 or left.dtype != right.dtype:
            return ComparisonResult(False, "batched_unordered_topk", {"error": "dtype/shape mismatch"}, failed_path=path)
        batch, width = left.shape
        if self.sampled:
            passed = left.value == right.value and left.contract() == right.contract()
            return ComparisonResult(
                passed, "sampled_unordered_topk",
                {"sample_count": len(left.value), "original_shape": left.shape,
                 "canonicalized_per_row": True},
                () if passed else ({"error": "sampled topk mismatch"},), None if passed else path,
            )
        if len(left.value) != batch * width or len(right.value) != batch * width:
            return ComparisonResult(False, "batched_unordered_topk",
                                    {"error": "small correctness payload is incomplete"},
                                    failed_path=path)
        failed = []
        for row in range(batch):
            a = tuple(int(v) for v in left.value[row * width : (row + 1) * width])
            b = tuple(int(v) for v in right.value[row * width : (row + 1) * width])
            if len(set(a)) != width or len(set(b)) != width or set(a) != set(b):
                failed.append({"batch": row, "reference": a[:8], "candidate": b[:8]})
        return ComparisonResult(
            not failed,
            "batched_unordered_topk",
            {"batch": batch, "k": width, "unordered": True, "failed_batches": len(failed)},
            tuple(failed[:8]),
            None if not failed else path,
        )


def _runtime():
    torch = importlib.import_module("torch")
    topk = importlib.import_module("sglang.jit_kernel.dsv4.topk")
    return torch, topk.plan_topk_v2


class DeepSeekV4TopKTransformSpec:
    operator_id = "deepseek_v4_topk_transform"

    def cases(self):
        return (
            CaseSpec("smoke_b2_l128_k8", {"batch": 2, "length": 128, "seq_len": 117, "topk": 8, "page_size": 64, "pattern": "random"}, 31, frozenset({"smoke"}), 600),
            CaseSpec("boundary_k1", {"batch": 2, "length": 68, "seq_len": 65, "topk": 1, "page_size": 64, "pattern": "random"}, 37, frozenset({"boundary"}), 600),
            CaseSpec("boundary_cutoff_tie", {"batch": 2, "length": 132, "seq_len": 129, "topk": 16, "page_size": 64, "pattern": "tie"}, 41, frozenset({"boundary", "adversarial"}), 600),
            CaseSpec("boundary_all_negative", {"batch": 2, "length": 132, "seq_len": 129, "topk": 8, "page_size": 64, "pattern": "negative"}, 43, frozenset({"boundary"}), 600),
            CaseSpec("boundary_tail_page", {"batch": 2, "length": 196, "seq_len": 191, "topk": 32, "page_size": 64, "pattern": "random"}, 47, frozenset({"boundary"}), 600),
            CaseSpec("representative_decode_b16_l65536_k1024", {"batch": 16, "length": 65536, "seq_len": 65536, "topk": 1024, "page_size": 64, "pattern": "random"}, 53, frozenset({"representative", "legacy", "decode", "performance_only"}), 1800),
        ) + _projection_cases()

    def legacy_mappings(self):
        return LEGACY_TOPK

    @staticmethod
    def _validate(symbols):
        batch, length, seq_len, topk, page_size = (int(symbols[n]) for n in ("batch", "length", "seq_len", "topk", "page_size"))
        query_tokens = int(symbols.get("query_tokens", 1))
        if batch <= 0 or query_tokens <= 0 or length <= 0 or topk <= 0 or topk > 2048:
            raise ValueError("batch/query_tokens/length/topk must be positive and topk <= 2048")
        if seq_len < 0 or seq_len > length:
            raise ValueError("seq_len must satisfy 0 <= seq_len <= length")
        if seq_len < topk:
            raise ValueError("formal cases require seq_len >= topk; kernel -1 padding is tested separately")
        if page_size <= 0 or page_size & (page_size - 1):
            raise ValueError("page_size must be a positive power of two")
        if symbols.get("phase") == "prefill" and batch != 1:
            raise ValueError("prefill CaseSpec must describe exactly one request")
        return batch, query_tokens, length, seq_len, topk, page_size

    def make_inputs(self, case, context):
        torch, plan = _runtime()
        batch, query_tokens, length, seq_len, topk, page_size = self._validate(case.symbols)
        rows = batch * query_tokens
        device = next(iter(context.cuda), "cuda:0")
        generator = context.cuda.get(device)
        if generator is None:
            raise RuntimeError("deepseek_v4_topk_transform requires a CUDA generator")
        scores = torch.randn((rows, length), device=device, dtype=torch.float32, generator=generator)
        pattern = case.symbols.get("pattern", "random")
        if pattern == "tie":
            scores.fill_(-4.0)
            scores[:, : topk + 3] = 1.0
        elif pattern == "negative":
            scores.copy_(-scores.abs() - 0.01)
        if query_tokens > 1:
            raw_context = int(case.symbols["raw_context"])
            raw_prefix = raw_context - query_tokens
            seq_lens = torch.div(
                torch.arange(raw_prefix + 1, raw_context + 1, device=device, dtype=torch.int32),
                4,
                rounding_mode="floor",
            )
        else:
            seq_lens = torch.full((rows,), seq_len, device=device, dtype=torch.int32)
        pages = (length + page_size - 1) // page_size
        request_tables = torch.arange(batch * pages, device=device, dtype=torch.int32).view(batch, pages)
        page_tables = request_tables[:, None, :].expand(batch, query_tokens, pages).reshape(rows, pages).contiguous()
        output = torch.empty((rows, topk), device=device, dtype=torch.int32)
        metadata = plan(seq_lens, static_threshold=0)
        return InputBundle(args=(scores, seq_lens, page_tables, output, page_size, metadata))

    def clone_inputs(self, inputs):
        scores, seq_lens, page_tables, output, page_size, metadata = inputs.args
        return InputBundle(args=(scores.clone(), seq_lens.clone(), page_tables.clone(), output.clone(), page_size, metadata.clone()))

    def normalize_output(self, output):
        torch = importlib.import_module("torch")
        # Canonicalize every row before bounded sampling so backend-specific
        # ordering of an otherwise identical TopK set cannot cause a failure.
        flat = torch.sort(output.detach(), dim=-1).values.reshape(-1)
        count = min(flat.numel(), 256)
        if count <= 1:
            indices = torch.zeros(count, device=flat.device, dtype=torch.long)
        else:
            positions = torch.arange(count, device=flat.device, dtype=torch.long)
            span, intervals = flat.numel() - 1, count - 1
            indices = positions * (span // intervals) + positions * (span % intervals) // intervals
        values = tuple(int(value) for value in flat[indices].cpu().tolist())
        return OutputBundle((OutputLeaf("output", values, str(output.dtype), tuple(output.shape),
                                       tuple(output.stride()), "strided", str(output.device)),))

    def comparator(self, case):
        return BatchedUnorderedTopKComparator(sampled="performance_only" in case.tags)

    def cost_model(self, case):
        batch, query_tokens, _, seq_len, topk, _ = self._validate(case.symbols)
        rows = batch * query_tokens
        # One auditable score comparison unit per valid score. The sort's exact
        # implementation-dependent comparison count is intentionally not
        # presented as hardware FLOPs.
        if query_tokens > 1:
            raw_context = int(case.symbols["raw_context"])
            work = sum(
                (raw_context - query_tokens + index + 1) // 4
                for index in range(query_tokens)
            )
        else:
            work = rows * seq_len
        return {"flops": work, "estimated_bytes": work * 4 + rows * topk * 4, "throughput_units": work}


SPEC = DeepSeekV4TopKTransformSpec()


def cost_model(case):
    return SPEC.cost_model(case)
