from __future__ import annotations

import importlib.util
from types import SimpleNamespace

import torch


SPEC_PATH = (
    "/home/gjy/data/agent4kernel/mega_mqa_logits/llm_flops/operators/"
    "references/deepseek_v4_mega_mqa_logits/spec.py"
)


def load_spec():
    module_spec = importlib.util.spec_from_file_location("mega_v2_spec", SPEC_PATH)
    module = importlib.util.module_from_spec(module_spec)
    assert module_spec.loader is not None
    module_spec.loader.exec_module(module)
    return module


def main():
    spec_module = load_spec()
    torch.cuda.set_device(0)
    for case in spec_module.SPEC.cases():
        generator = torch.Generator(device="cuda:0")
        generator.manual_seed(case.seed)
        bundle = spec_module.SPEC.make_inputs(
            case, SimpleNamespace(cuda={"cuda:0": generator})
        )
        (
            hidden_states,
            _,
            c4_state,
            _,
            _,
            rope_cos,
            _,
            q_fp8,
            _,
            kv_fused,
            context_lens,
            page_table,
            _,
            cache_write_locs,
            raw_context,
            compressed_context,
            _,
        ) = bundle.args
        m = case.symbols["m"]
        prefix = raw_context - m
        expected_lens = (
            torch.arange(
                prefix + 1, raw_context + 1, device="cuda:0", dtype=torch.int32
            )
            // 4
        )
        assert hidden_states.shape == (m, 7168)
        assert q_fp8.shape == (m, 1, 64, 128)
        assert c4_state.shape == (8, 512)
        assert rope_cos.shape == (m // 4, 32)
        assert kv_fused.shape == (256, 64, 1, 132)
        assert page_table.shape == (m, 256)
        assert torch.equal(page_table, page_table[:1].expand_as(page_table))
        assert torch.unique(page_table).numel() == 256
        assert torch.equal(context_lens[:, 0], expected_lens)
        assert cache_write_locs.numel() == m // 4
        assert cache_write_locs[0].item() == prefix // 4
        assert cache_write_locs[-1].item() == compressed_context - 1
        print(
            case.case_id,
            {
                "requests": 1,
                "queries": m,
                "physical_pages": kv_fused.shape[0],
                "context_first": context_lens[0, 0].item(),
                "context_last": context_lens[-1, 0].item(),
                "new_c4_k": cache_write_locs.numel(),
            },
        )


if __name__ == "__main__":
    main()
