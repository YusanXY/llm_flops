# CSV schemas

These schemas are identical under repository `results/` and KDA task `bench/`.
Generated CSV files are recovery artifacts owned by the engine and must not be
hand-edited by an Agent.

`benchmark_engine.reporting.csv_writer` is authoritative. CSV files are UTF-8
RFC 4180 with CRLF records. `schema_version` is first; empty is the only null;
booleans are lowercase; numbers must be finite. Writers validate the complete
row, fsync a same-directory temporary file, and atomically replace the table.
Identical primary-key rows are idempotent; different content is a conflict.

## Versions

- `results.csv`: v4;
- `performance_samples.csv`: v3;
- correctness, projection, run index, and history: v1.

Valid results v1/v2/v3 and samples v1/v2 are read through explicit migration.
The next append atomically writes the current header. Results migrated from
v1-v3 receive `imported_legacy=false`; old missing gate provenance remains
non-rankable. Unknown headers or enum values are rejected.

## results.csv v4

One row summarizes one operator/candidate/case/seed; primary key `result_id`.
Stable groups are:

- identity: run/evaluation/result/operator/candidate/reference IDs, contract,
  source hashes, environment, case/hash/seed/tags/input summary;
- outcomes: overall, correctness, performance, skip/error/diagnostic fields;
- timers: requested/effective/fallback plus import, build, first call, warmup,
  graph capture, steady-state, and reference equivalents;
- statistics: mean/median/min/max/percentiles/population stddev/CV per role;
- fairness/gates: formal/ranking flags, reasons, resolved thresholds, telemetry;
- cost and memory: FLOPs, bytes, arithmetic intensity, throughput, allocated,
  reserved, and workspace bytes;
- `legacy_graph_ms`: the only aggregate latency admitted from the old
  `graph_ms` CSV path.

Closed status enums are defined in `benchmark_engine.models`. `passed` means
all requested formal gates passed; `skipped` is policy/non-execution;
`unsupported`, `error`, `timeout`, `oom`, and `crashed` remain distinct.

### imported_legacy contract

`imported_legacy` is required. For normal rows it is `false`, and timestamp,
reference ID, source hashes, and case hash remain mandatory through row-level
validation; `legacy_graph_ms` must be empty.

For `true` rows the v4 validator, on append, append-many, read, and migration
output, requires:

- suite `legacy_import`, mode `performance`;
- overall/correctness/performance all `skipped` with the fixed import reason;
- implementation source hashes, reference ID, case hash, timestamp empty;
- formal/ranking false and performance gate skipped;
- no correctness metrics, stage timings, raw-derived statistics/CV, speedup,
  cost model, memory/workspace, GPU/toolchain claims, or diagnostics;
- executed legacy data has one finite non-negative `legacy_graph_ms`;
  unavailable legacy data has no latency and `error_type=legacy_unavailable`.

The old `graph_ms` value is one CUDA Graph replay average per call. It is not
renamed to median and never creates `performance_samples.csv` rows.

## correctness_outputs.csv v1

Primary key `(result_id, output_path)`. It records comparator, reference and
candidate dtype/shape, tolerance, pass flag, error metrics, NaN/mismatch counts,
and diagnostic path. Legacy imports contain only the header.

## performance_samples.csv v3

Primary key `(result_id, implementation_role, sample_index)`. Every row records
reference/candidate role, both normalized dtype/shape maps, sample and inner
iteration counts, elapsed/per-call milliseconds, global `order_index`, timer
selection/fallback, and optional telemetry. `order_index` is execution order,
not performance rank. Legacy imports contain only the header.

## model_projection.csv v1

Primary key `(result_id, adapter_id, implementation_role)`. Projection keeps
phase/profile/model input/context, adapter/backend/kind/legacy shape/instances,
per-call latency, projected model latency, status, and reason. For legacy
imports, `legacy_shape` is a stable JSON object containing the exact source CSV
`input_shape` and `output_shape`, plus nullable current-mapping `logical_shape`.
Formal engine
projection derives from candidate median. Legacy import projection preserves
the old aggregate as explicitly non-formal data.

## Indexes

`run_index.csv` maps a run to mirrored evaluations. Candidate `history.csv`
and `latest.json` publish complete normal evaluations. Legacy imports write
none of these, so run summarize/resume/compare cannot discover them.
