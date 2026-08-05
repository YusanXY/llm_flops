"""DeepSeek-V3.2 DSA contest contract for the official FlashInfer baseline."""

import importlib

from benchmark_engine.correctness import InputBundle
from benchmark_engine.correctness.models import (
    ComparisonResult,
    OutputBundle,
    OutputLeaf,
)
from benchmark_engine.models import CaseSpec


HEADS = 16
CKV_DIM = 512
KPE_DIM = 64
PAGE_SIZE = 64
TOPK = 2048
SM_SCALE = 0.1352337788608801
OFFICIAL_NUM_PAGES = 8462
TOLERANCE = 0.2


def _torch():
    return importlib.import_module("torch")


def _case(case_id, num_tokens, num_pages, valid_topk, tags):
    return CaseSpec(
        case_id,
        {
            "num_tokens": num_tokens,
            "num_pages": num_pages,
            "num_qo_heads": HEADS,
            "head_dim_ckv": CKV_DIM,
            "head_dim_kpe": KPE_DIM,
            "page_size": PAGE_SIZE,
            "topk": TOPK,
            "valid_topk": valid_topk,
        },
        532,
        frozenset(tags),
        1800,
    )


def _sample_indices(torch, total, device, limit=256):
    count = min(int(total), int(limit))
    if count == 0:
        return torch.empty(0, device=device, dtype=torch.long)
    if count == 1:
        return torch.zeros(1, device=device, dtype=torch.long)
    positions = torch.arange(count, device=device, dtype=torch.long)
    span, intervals = int(total) - 1, count - 1
    return positions * (span // intervals) + positions * (span % intervals) // intervals


def _semantic_oracle(torch, q_nope, q_pe, ckv_cache, kpe_cache, indices):
    ckv = ckv_cache.reshape(-1, CKV_DIM).float()
    kpe = kpe_cache.reshape(-1, KPE_DIM).float()
    invalid = indices == -1
    safe = indices.masked_fill(invalid, 0).long()
    selected_ckv = ckv[safe]
    selected_kpe = kpe[safe]
    logits = (
        torch.einsum("thd,tkd->thk", q_nope.float(), selected_ckv)
        + torch.einsum("thd,tkd->thk", q_pe.float(), selected_kpe)
    ) * SM_SCALE
    logits.masked_fill_(invalid.unsqueeze(1), float("-inf"))
    probabilities = torch.softmax(logits, dim=-1)
    return torch.einsum("thk,tkd->thd", probabilities, selected_ckv)


def _bounded_output(output, limit=256):
    torch = _torch()
    if isinstance(output, (tuple, list)):
        if not output:
            raise RuntimeError("FlashInfer baseline returned an empty sequence")
        output = output[0]
    flat = output.detach().reshape(-1)
    indices = _sample_indices(torch, flat.numel(), flat.device, limit)
    return OutputBundle((OutputLeaf(
        "output",
        tuple(flat[indices].float().cpu().tolist()),
        str(output.dtype),
        tuple(output.shape),
        tuple(output.stride()),
        "strided",
        str(output.device),
    ),))


class DsaComparator:
    def compare(self, reference, candidate, **_):
        failures, maxima = [], {}
        for role, bundle in (("reference", reference), ("candidate", candidate)):
            leaves = bundle.by_path()
            output = leaves.get("output")
            oracle = leaves.get("state.semantic_oracle")
            if output is None or oracle is None:
                failures.append({"role": role, "error": "missing output/oracle"})
                continue
            maximum = max(
                (abs(float(a) - float(b)) for a, b in zip(output.value, oracle.value)),
                default=0.0,
            )
            maxima[role] = maximum
            if len(output.value) != len(oracle.value) or maximum > TOLERANCE:
                failures.append({"role": role, "max_abs_error": maximum})

        left, right = reference.by_path(), candidate.by_path()
        left_output, right_output = left.get("output"), right.get("output")
        if left_output is not None and right_output is not None:
            cross = max(
                (abs(float(a) - float(b)) for a, b in zip(left_output.value, right_output.value)),
                default=0.0,
            )
            maxima["reference_candidate"] = cross
            if len(left_output.value) != len(right_output.value) or cross > TOLERANCE:
                failures.append({"path": "output", "max_abs_error": cross})

        for path in sorted(set(left) | set(right)):
            a, b = left.get(path), right.get(path)
            if a is None or b is None or a.contract() != b.contract():
                failures.append({"path": path, "error": "contract mismatch"})
            elif path.startswith("state.") and path != "state.semantic_oracle" and a.value != b.value:
                failures.append({"path": path, "error": "observed input state mutated"})

        return ComparisonResult(
            not failures,
            "dsv32_dsa_oracle_and_state",
            {"max_abs_error": maxima},
            tuple(failures[:8]),
            None if not failures else "output",
        )


class DeepSeekV32DsaSparseAttentionSpec:
    operator_id = "deepseek_v32_dsa_sparse_attention"

    def cases(self):
        # The official contest JSONL has 23 workloads but only these five
        # distinct tensor geometries; all use num_pages=8462.
        official = tuple(
            _case(
                f"mlsys26_tokens{tokens}_pages8462_topk2048",
                tokens,
                OFFICIAL_NUM_PAGES,
                TOPK,
                {"official", "mlsys26", "representative", "decode", f"tokens_{tokens}"},
            )
            for tokens in (1, 2, 6, 7, 8)
        )
        return (
            _case(
                "smoke_tokens1_pages32_topk2048",
                1,
                32,
                TOPK,
                {"smoke", "boundary", "decode"},
            ),
            _case(
                "padding_tokens2_pages64_valid1024",
                2,
                64,
                1024,
                {"boundary", "padding", "decode"},
            ),
        ) + official

    @staticmethod
    def _validate(symbols):
        values = tuple(
            int(symbols[name])
            for name in (
                "num_tokens", "num_pages", "num_qo_heads", "head_dim_ckv",
                "head_dim_kpe", "page_size", "topk", "valid_topk",
            )
        )
        tokens, pages, heads, ckv_dim, kpe_dim, page_size, topk, valid = values
        if tokens <= 0 or pages <= 0 or not 0 < valid <= topk:
            raise ValueError("token/page/topk geometry is invalid")
        if (heads, ckv_dim, kpe_dim, page_size, topk) != (
            HEADS, CKV_DIM, KPE_DIM, PAGE_SIZE, TOPK
        ):
            raise NotImplementedError("unsupported DeepSeek-V3.2 contest layout")
        if pages * page_size < valid:
            raise ValueError("KV cache has fewer physical tokens than valid_topk")
        return tokens, pages, valid

    def layout_contract(self, case):
        tokens, pages, valid = self._validate(case.symbols)
        return {
            "q_nope": [tokens, HEADS, CKV_DIM],
            "q_pe": [tokens, HEADS, KPE_DIM],
            "ckv_cache": [pages, PAGE_SIZE, CKV_DIM],
            "kpe_cache": [pages, PAGE_SIZE, KPE_DIM],
            "sparse_indices": [tokens, TOPK],
            "valid_topk": valid,
            "cache_mutation": "forbidden",
            "workspace_bytes": 128 * 1024 * 1024,
        }

    def make_inputs(self, case, context):
        tokens, pages, valid = self._validate(case.symbols)
        torch = _torch()
        device = next(iter(context.cuda), "cuda:0")
        generator = context.cuda.get(device)
        if generator is None:
            raise RuntimeError("DeepSeek-V3.2 DSA reference requires a CUDA generator")

        q_nope = torch.randn(
            (tokens, HEADS, CKV_DIM), device=device, dtype=torch.bfloat16,
            generator=generator,
        )
        q_pe = torch.randn(
            (tokens, HEADS, KPE_DIM), device=device, dtype=torch.bfloat16,
            generator=generator,
        )
        ckv_cache = torch.randn(
            (pages, PAGE_SIZE, CKV_DIM), device=device, dtype=torch.bfloat16,
            generator=generator,
        )
        kpe_cache = torch.randn(
            (pages, PAGE_SIZE, KPE_DIM), device=device, dtype=torch.bfloat16,
            generator=generator,
        )

        total_tokens = pages * PAGE_SIZE
        starts = torch.randint(
            0, total_tokens, (tokens, 1), device=device, dtype=torch.int64,
            generator=generator,
        )
        offsets = torch.arange(valid, device=device, dtype=torch.int64).unsqueeze(0)
        selected = ((starts + offsets * 97) % total_tokens).to(torch.int32)
        sparse_indices = torch.full(
            (tokens, TOPK), -1, device=device, dtype=torch.int32
        )
        sparse_indices[:, :valid] = selected

        oracle = _semantic_oracle(
            torch, q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices
        )
        oracle_flat = oracle.reshape(-1)
        oracle_indices = _sample_indices(torch, oracle_flat.numel(), device)
        sampled_oracle = oracle_flat[oracle_indices]
        observed = {
            "semantic_oracle": sampled_oracle,
            "q_nope_head": q_nope[:1, :1, :8],
            "q_pe_head": q_pe[:1, :1, :8],
            "ckv_cache_head": ckv_cache[:1, :1, :8],
            "kpe_cache_tail": kpe_cache[-1:, -1:, :8],
            "indices": sparse_indices,
        }
        return InputBundle(
            args=(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, SM_SCALE),
            observed_state=observed,
        )

    def clone_inputs(self, inputs):
        q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, scale = inputs.args
        qn2, qp2 = q_nope.clone(), q_pe.clone()
        ckv2, kpe2 = ckv_cache.clone(), kpe_cache.clone()
        indices2 = sparse_indices.clone()
        observed = {
            "semantic_oracle": inputs.observed_state["semantic_oracle"].clone(),
            "q_nope_head": qn2[:1, :1, :8],
            "q_pe_head": qp2[:1, :1, :8],
            "ckv_cache_head": ckv2[:1, :1, :8],
            "kpe_cache_tail": kpe2[-1:, -1:, :8],
            "indices": indices2,
        }
        return InputBundle(
            args=(qn2, qp2, ckv2, kpe2, indices2, scale),
            observed_state=observed,
        )

    def normalize_output(self, output):
        return _bounded_output(output)

    def comparator(self, case):
        self._validate(case.symbols)
        return DsaComparator()

    def cost_model(self, case):
        tokens, _, valid = self._validate(case.symbols)
        flops = 2 * tokens * HEADS * valid * (CKV_DIM + KPE_DIM + CKV_DIM)
        estimated_bytes = 2 * (
            tokens * HEADS * (CKV_DIM + KPE_DIM)
            + tokens * valid * (CKV_DIM + KPE_DIM)
            + tokens * HEADS * CKV_DIM
        ) + tokens * TOPK * 4
        return {
            "flops": flops,
            "estimated_bytes": estimated_bytes,
            "throughput_units": tokens * valid,
        }

    def workspace_bytes(self, case):
        self._validate(case.symbols)
        return 128 * 1024 * 1024

    def legacy_mappings(self):
        return {
            "source": "flashinfer-ai/mlsys26-contest",
            "definition": "dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64",
            "solution": "flashinfer_wrapper_5af199",
            "hardware": "NVIDIA B200",
            "official_num_pages": OFFICIAL_NUM_PAGES,
            "official_num_tokens": (1, 2, 6, 7, 8),
        }


SPEC = DeepSeekV32DsaSparseAttentionSpec()


def cost_model(case):
    return SPEC.cost_model(case)
