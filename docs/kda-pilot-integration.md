# KDA-Pilot task-native integration

KDA-Pilot task mode evaluates source where the Agent works. It removes the old
stage-copy step and gives the task one authoritative baseline, versioned
solutions, task-local caches, and task-local artifacts.

## Ownership model

| Path | Owner | Mutability |
|---|---|---|
| `baseline/` | evaluation administrator | immutable to Agents |
| `solution/<candidate_id>/` | Agent | mutable until measured; version instead of overwriting evidence |
| `bench/` | llm_flops artifact writer | generated; never hand-edit CSV files |
| `.cache/llm-flops/` | compiler/JIT runtime | disposable and never an artifact |
| `docs/`, `profile/`, `ncu/`, `tests/` | KDA task | supporting evidence only |

The task root must contain `baseline/`, `solution/`, and `bench/`. The baseline
contains the same strict contract files as repository mode:

```text
baseline/
  operator.yaml
  spec.py
  implementation.py
  README.md
```

The Registry reads those files directly. It rejects a symlinked baseline or
solution directory and rejects flat `solution/implementation.py` or
`solution/candidate.yaml`; every candidate must have its own immediate version
directory. Discovery remains static and never imports candidate code.

## Install migrated contracts

The administrative migration mapping is
`integrations/kda-pilot/operators.json`. It maps every migrated non-tutorial
operator to one unique KDA task slug. Populate or refresh a KDA workload tree:

```bash
.runtime/venv/bin/python tools/install_kda_workload.py \
  --llm-root /path/to/KDA-Pilot/llm \
  --refresh-baseline \
  --rewrite-task-guides
```

This command is for deployment owners, not optimization Agents. It copies the
reviewed baseline contracts and existing control candidates once, writes task
guidance, and refuses to overwrite a differing solution version. Normal
benchmark execution never calls this installer and never copies source back to
`operators/`.

Audit every mapped task without modifying it:

```bash
.runtime/venv/bin/python tools/install_kda_workload.py \
  --llm-root /path/to/KDA-Pilot/llm --verify-only
```

`example_cpu_add` intentionally remains a repository-only tutorial and is not
installed as an LLM optimization task.

## One-command Agent interface

From a task root:

```bash
CUDA_VISIBLE_DEVICES=0 ../../../llm_flops/kda-bench.sh . <candidate_id>
```

The wrapper performs:

1. `bench validate --task-root ...`;
2. `bench run --task-root ... --suite kda_task --candidate ...`;
3. correctness gating followed by formal performance when correctness passes;
4. artifact publication below the task's `bench/` directory.

Extra arguments are passed to `bench run`:

```bash
../../../llm_flops/kda-bench.sh . optimized_v3 \
  --case 'm4096*' --samples 50 --inner-iterations 20
```

The explicit CLI remains available:

```bash
./bench.sh list --task-root /path/to/task
./bench.sh validate --task-root /path/to/task
./bench.sh run --task-root /path/to/task --suite kda_task \
  --mode correctness --candidate optimized_v3
./bench.sh run --task-root /path/to/task --suite kda_task \
  --mode performance --candidate optimized_v3
```

Without `--output-root`, task mode writes to `<task-root>/bench`. An explicit
output root is still supported for controlled experiments. Repository mode is
unchanged and continues to use `operators/references`, `operators/candidates`,
and `results`.

## Runtime and cache consistency

`bench.sh` separates the engine checkout from its dependency runtime:

- `BENCHMARK_RUNTIME_ROOT` selects a venv created from
  `requirements/benchmark-lock.json`; it defaults to `<llm_flops>/.runtime`.
- `BENCHMARK_CACHE_ROOT` selects UV, Torch extension, FlashInfer, and XDG cache
  roots; `kda-bench.sh` defaults it to `<task>/.cache/llm-flops`.
- `PYTHONPATH` is set to the current checkout's `src/`, so a reused runtime
  cannot silently import an editable install from another llm_flops checkout.

Agents must not install packages, recreate the runtime, edit the lock, or swap
CUDA/toolchain components. If the lock check fails, stop and ask the deployment
owner to provision a matching runtime. Compiler/JIT/build time is recorded
separately and is not included in steady-state samples.

## Result layout

Task mode retains the normal mirror below `bench/`:

```text
bench/<operator_id>/<candidate_id>/<evaluation_id>/
  evaluation_manifest.json
  results.csv
  correctness_outputs.csv
  performance_samples.csv
  model_projection.csv
  summary.md
  diagnostics/
  logs/
```

The manifest records absolute baseline and candidate source hashes indirectly
through their content digests, the environment fingerprint, resolved policy,
command, and lifecycle state. CSV semantics are identical to repository mode;
see [CSV schema](csv-schema.md).

`order_index` in `performance_samples.csv` is the deterministic A/B execution
order, not a ranking. Ranking requires passed correctness, formal performance,
a passed performance gate, and `ranking_eligible=true`. A timeout, crash, OOM,
unsupported path, high coefficient of variation, or competing GPU process
remains visible and cannot become a ranked result by omission.

## Agent safety rules

- Never modify `baseline/` or any evaluation policy it imports.
- Never flatten multiple attempts into one `solution/` directory.
- Never write a task-local benchmark implementation or correctness oracle.
- Never tune against changed cases, seeds, tolerances, gates, or output shapes.
- Never compare results with different source hashes or environment identity as
  if they were one formal A/B run.
- Never hand-edit generated CSV, manifest, summary, or diagnostics files.
- Use one idle target GPU; do not overlap another benchmark or profiler.

These rules keep the Agent's editable surface small while preserving the
controller/worker isolation, correctness gate, fair timing, resume, raw sample,
and artifact contracts of the benchmark engine.
