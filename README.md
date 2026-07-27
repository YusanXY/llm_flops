# LLM operator benchmark engine

This repository evaluates LLM operator implementations with reproducible
correctness gates, isolated workers, staged timing, raw samples, and mirrored
artifacts. An optimized implementation such as DeepGEMM or FlashInfer is the
`reference`; code under evaluation is a `candidate`. A byte-identical copy of
the reference is a useful control candidate when validating the framework.

This branch is the MI300X/ROCm adaptation of the engine. PyTorch continues to
expose HIP events and graphs through the `torch.cuda` namespace, so the stable
`cuda_event` and `cuda_graph` CLI timer names are retained while result rows
record `accelerator_backend=rocm`. The original DeepSeek and GLM-5 launchers
remain available as legacy baselines but are not MI300X release gates.

## Install

Run from the repository root on the configured MI300X host. Python 3.11,
PyTorch 2.10.0+rocm7.0, SGLang, and AITER are provided by the shared
environment named in `~/.config/agent4kernel/env.sh`:

```bash
./bootstrap.sh
./run.sh check
./bench.sh --help
./bench.sh env
```

`bootstrap.sh` verifies the immutable environment lock and creates the safe
`.runtime/venv` symlink to `$GPU_VENV`; it does not download or reinstall
large GPU dependencies. Both launchers discard inherited `PYTHONPATH` and
explicitly select the llm_flops, SGLang, and AITER source trees. See the
[MI300X runtime guide](docs/mi300x.md).

## KDA-Pilot task mode

The recommended Agent integration evaluates sources directly in one
`KDA-Pilot/llm/<task>` directory:

```text
baseline/                         immutable reference contract
solution/<candidate_id>/          one independent candidate version
bench/<operator>/<candidate>/<evaluation>/
```

From the task directory, one command performs static validation, correctness,
and gated performance, then publishes results under `bench/`:

```bash
ROCR_VISIBLE_DEVICES=0 ../../../llm_flops/kda-bench.sh . <candidate_id>
```

No source is staged into this repository. Agents must not modify `baseline/`,
the environment lock, cases, comparators, tolerances, timers, or gates. See the
[KDA-Pilot integration contract](docs/kda-pilot-integration.md).
This branch targets MI300X (`gfx942`) and uses
`KDA-Pilot/external/ROCm-KernelWiki-Q` as its kernel knowledge base.

## Five-minute CPU check

`example_cpu_add` exercises discovery, Controller/Worker isolation,
correctness, wall-clock performance, and artifact writing without CUDA:

```bash
./bench.sh validate --operator example_cpu_add
./bench.sh list --operator example_cpu_add
./bench.sh run --suite smoke --operator example_cpu_add
./bench.sh run --suite smoke --operator example_cpu_add --dry-run
```

The real run prints a `run_id` and mirrored evaluation paths. Read an artifact
without importing candidate code:

```bash
./bench.sh summarize results/OPERATOR/CANDIDATE/EVALUATION
```

## Reference and candidate layouts

Repository mode remains available for framework development and regression:

```text
operators/references/<operator_id>/
  operator.yaml
  spec.py
  implementation.py
operators/candidates/<operator_id>/<candidate_id>/
  implementation.py
```

`operator.yaml` is the static contract. `spec.py` creates deterministic cases,
isolated inputs, output normalization/comparison, and an optional cost model.
`implementation.py` exports the timed callable. A candidate may optionally add
`candidate.yaml`. Candidate IDs only need to be unique under one operator and
path-safe; the task/timestamp/hash form is a recommendation, not a requirement.

KDA task mode uses the same contract files without copying them at benchmark
time:

```text
baseline/{operator.yaml,spec.py,implementation.py,README.md}
solution/<candidate_id>/implementation.py
```

Use `--task-root /path/to/task` with `list`, `validate`, `run`, `summarize`, or
`compare`. If `--output-root` is omitted, task-mode results go to
`<task-root>/bench`; repository mode still defaults to `results/`.

See [the operator contract](docs/operator-contract.md) and
[candidate guide](docs/candidate-guide.md) before adding an implementation.

## Run modes

Correctness runs the reference and candidate on isolated clones:

```bash
./bench.sh run --mode correctness \
  --operator example_cpu_add \
  --candidate quickstart__20260716T120000Z__4279e756 \
  --case tiny --seed 0
```

Performance still applies the correctness gate first. JIT/build, first call,
warmup, graph capture, and steady-state samples remain separate:

```bash
./bench.sh run --mode performance \
  --operator example_cpu_add \
  --candidate quickstart__20260716T120000Z__4279e756 \
  --case tiny --timer wall_clock \
  --warmup 5 --samples 30 --inner-iterations 20
```

`all` requests correctness and formal performance in one evaluation:

```bash
ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 \
  ./bench.sh run --suite mi300x_smoke
```

Resume and compare only formal engine evaluations:

```bash
./bench.sh run --resume RUN_ID
./bench.sh compare --run RUN_ID --baseline-run BASELINE_RUN_ID
```

## Read results

Repository-mode evaluations use `results/`; KDA task mode uses `bench/`. Both
retain the same identity mirror and artifact schema:

```text
results/<operator_id>/<candidate_id>/<evaluation_id>/
  evaluation_manifest.json
  results.csv
  correctness_outputs.csv
  performance_samples.csv
  model_projection.csv
  summary.md
  diagnostics/
  logs/
```

`results.csv` is the case summary. `correctness_outputs.csv` contains normalized
leaf comparisons. `performance_samples.csv` contains every reference/candidate
sample, dtype/shape contract, and execution `order_index` (`order_index` is not
a ranking). `model_projection.csv` multiplies measured per-call latency by
declared model instances without changing raw results.

Only rows with passed correctness, formal performance, a passed performance
gate, and `ranking_eligible=true` may rank. System overlap or high CV leaves
diagnostic samples visible but non-rankable.

## Legacy results and regression

Import a legacy DeepSeek V4 **FP8/MXFP8** CSV without inventing source hashes,
correctness, samples, CV, median, speedup, or ranking eligibility:

```bash
.runtime/venv/bin/python tools/convert_legacy_results.py \
  results/deepseek_v4_pro_fp8_mxfp8_prefill_kv65536_m1024.csv \
  --candidate-id legacy_control --dry-run
```

Remove `--dry-run` to publish strict `legacy_import_manifest.json` artifacts.
Legacy imports are not added to `run_index.csv`; resume and compare reject them.
MXFP4 CSV import is deliberately unsupported and fails with a stable error.

The predeclared B200 comparison is retained only for historical result
compatibility and is not part of the MI300X gate:

```bash
.runtime/venv/bin/python tools/regress_deepseek_v4.py \
  --phase prefill --work-dir .runtime/regression/prefill --dry-run
```

The execution form removes `--dry-run`. Thresholds and evidence requirements
are fixed in [the migration guide](docs/migration-llm-flops.md).

## GLM-5 operators

The five legacy single-operator entries and both unified GLM-5 single-GPU
scripts are available through discoverable contracts:

```bash
./bench.sh validate --operator 'glm5_*'
CUDA_VISIBLE_DEVICES=0 ./bench.sh run --suite glm5_smoke
CUDA_VISIBLE_DEVICES=0 ./bench.sh run --suite glm5_regression
```

The prefill/decode mappings cover all thirteen legacy rows. DeepEP is visible
as `glm5_deepep_dispatch` but explicitly unsupported because the legacy script
requires eight GPUs and distributed collective state; it is not included in
single-GPU suites or ranking. See the [GLM-5 migration and coverage
matrix](docs/glm5-migration.md).

Legacy sparse attention and contiguous MoE use separate operator IDs from the
unified-script sparse and masked-MoE rows. This preserves their different CUDA
graph/event and inner-iteration timing contracts; suites do not override those
per-operator settings.

## Exit codes

- `0`: command completed and every requested gate passed;
- `1`: evaluation completed but correctness/performance gate failed;
- `2`: usage, selector, registry, compatibility, or validation error;
- `3`: worker protocol or artifact infrastructure failure;
- `130`: user interruption after worker cleanup.

## Documentation

- [Documentation index](docs/index.md)
- [MI300X/ROCm runtime](docs/mi300x.md)
- [KDA-Pilot task-native integration](docs/kda-pilot-integration.md)
- [Getting started](docs/getting-started.md)
- [CLI](docs/cli.md)
- [Architecture](docs/architecture.md) and [implementation](docs/implementation.md)
- [Correctness](docs/correctness.md) and [performance](docs/performance.md)
- [CSV schema](docs/csv-schema.md) and [result layout](docs/result-layout.md)
- [Legacy migration and archived B200 regression](docs/migration-llm-flops.md)
- [GLM-5 migration and coverage](docs/glm5-migration.md)
- [Test tiers](docs/testing.md) and [troubleshooting](docs/troubleshooting.md)
- [Legacy launchers](docs/legacy-launchers.md)

Profiler integration and multi-GPU scheduling are outside the current engine
contract and are never silently represented as available.
