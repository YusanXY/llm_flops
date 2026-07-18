# CLI reference

Use `./bench.sh`; it selects the repository-local Python environment and cache
paths. All selectors are case-sensitive shell globs.

For a KDA task, `--task-root PATH` switches discovery from repository
`operators/` to `PATH/baseline` and `PATH/solution/<candidate_id>`. It is
supported by `list`, `validate`, `run`, `summarize`, and `compare`.

## Discovery and validation

```bash
./bench.sh list
./bench.sh list --operator 'deepseek_v4_*' --candidate 'reference_control__*'
./bench.sh validate
./bench.sh validate --operator deepseek_v4_fp8_gemm_nt
./bench.sh env --json
./bench.sh validate --task-root /path/to/KDA-Pilot/llm/task
./bench.sh list --task-root /path/to/KDA-Pilot/llm/task
```

`list` and `validate` are static: they do not import implementations or
initialize CUDA. Registry/no-match errors exit 2.

## Run

```text
bench run [--suite ID] [--mode all|correctness|performance]
          [--task-root KDA_TASK]
          [--operator GLOB ...] [--candidate GLOB ...]
          [--case GLOB ...] [--tag TAG ...] [--seed N ...]
```

Examples:

```bash
./bench.sh run --suite smoke --dry-run
./bench.sh run --mode correctness --operator OP --candidate CANDIDATE --case CASE
./bench.sh run --mode performance --operator OP --candidate CANDIDATE \
  --timer cuda_graph --warmup 5 --samples 20 --inner-iterations 1
./bench.sh run --mode all --operator OP --candidate CANDIDATE
./bench.sh run --task-root /path/to/task --suite kda_task --candidate CANDIDATE
```

Repeated values are ORed within a selector category and ANDed across
categories; exclusions apply last. CLI values override suite values, which
override operator defaults.

Important controls:

- `--dry-run`: emit a deterministic plan without formal environment/CUDA/result side effects;
- `--fail-fast`: stop scheduling after a failed result while retaining completed artifacts;
- `--timeout-s`: override worker stage timeout;
- `--task-root PATH`: use KDA `baseline/` and versioned `solution/` in place;
- `--output-root PATH`: replace `results/` in repository mode or `bench/` in task mode;
- `--evaluation-id ID`: explicit identity for one precisely selected candidate;
- `--resume RUN_ID`: verify manifest compatibility and execute missing result IDs only;
- `--timer auto|cuda_event|cuda_graph|wall_clock`;
- `--warmup`, `--samples`, `--inner-iterations`;
- performance gates: `--max-slowdown-pct`, `--min-speedup`,
  `--max-candidate-median-ms`, `--max-cv`, `--max-memory-bytes`;
- `--unsupported-policy fail|allow`, `--gpu-lock-timeout-s`;
- `--perf-on-correctness-fail`: retain explicitly non-formal diagnostic samples only.

The Agent-facing one-command wrapper validates and runs correctness plus
performance with task-local artifacts:

```bash
CUDA_VISIBLE_DEVICES=0 ../../../llm_flops/kda-bench.sh . CANDIDATE
```

See [KDA-Pilot integration](kda-pilot-integration.md) for runtime ownership,
baseline immutability, and result interpretation.

The terminal `results: N passed, M failed` line counts complete case/seed rows,
including performance gates. Use `summary.md` or `summarize` for separate
correctness, performance, and ranking-eligibility counts.

`deepseek_v4_prefill` covers M=1024/2048/4096; `deepseek_v4_decode`
covers batch 16/32. Both use raw context 65536.

```bash
CUDA_VISIBLE_DEVICES=0 ./bench.sh run --suite deepseek_v4_prefill
CUDA_VISIBLE_DEVICES=0 ./bench.sh run --suite deepseek_v4_decode
```

`glm5_smoke` selects bounded correctness cases for every supported single-GPU
GLM-5 family. `glm5_regression` adds formal performance sampling. The quoted
glob is expanded by the benchmark selector rather than the shell:

```bash
./bench.sh validate --operator 'glm5_*'
CUDA_VISIBLE_DEVICES=0 ./bench.sh run --suite glm5_smoke
CUDA_VISIBLE_DEVICES=0 ./bench.sh run --suite glm5_regression
```

`glm5_deepep_dispatch` validates and lists normally, but its cases are explicit
multi-GPU `unsupported` entries and are not selected by these single-GPU
suites. The GLM regression suite inherits each operator's timer and graph
inner-iteration settings, including the separate legacy/unified sparse and MoE
contracts. See [GLM-5 migration](glm5-migration.md).

## Summarize and compare

```bash
./bench.sh summarize results/OP/CANDIDATE/EVALUATION
./bench.sh summarize --operator OP --candidate CANDIDATE --evaluation EVALUATION
./bench.sh summarize --run RUN_ID
./bench.sh compare --result CURRENT --baseline-result BASELINE
./bench.sh compare --run CURRENT_RUN --baseline-run BASELINE_RUN
```

Summarize is read-only. Run-level forms resolve only `run_index.csv` entries.
Compare requires compatible contract/case/seed/source/environment/effective
timer and rankable inputs. A direct legacy-import summary is explicitly marked
non-formal; direct compare rejects it.

## Legacy tools

Legacy import and regression are explicit tools, not normal engine runs:

```bash
.runtime/venv/bin/python tools/convert_legacy_results.py INPUT.csv \
  --candidate-id legacy_control --dry-run
.runtime/venv/bin/python tools/regress_deepseek_v4.py \
  --phase prefill --work-dir .runtime/regression/prefill --dry-run
```

See [migration](migration-llm-flops.md) before removing `--dry-run`.

## Exit codes

- 0: requested gates passed;
- 1: completed correctness/performance gate failure;
- 2: usage/configuration/registry/compatibility error;
- 3: worker protocol or artifact infrastructure failure;
- 130: Ctrl-C after cleanup.
