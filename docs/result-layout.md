# Result layout and recovery

Candidate source and normal result paths share `<operator_id>/<candidate_id>`:

```text
operators/candidates/<operator_id>/<candidate_id>/
results/<operator_id>/<candidate_id>/<evaluation_id>/
```

KDA task mode keeps the same identity mirror while replacing the repository
source/result roots:

```text
solution/<candidate_id>/
bench/<operator_id>/<candidate_id>/<evaluation_id>/
```

`bench/` is selected automatically when `--task-root` is present. All files,
schemas, lifecycle rules, resume behavior, and ranking gates below are
identical; examples use the repository-mode name `results/` as a placeholder
for either artifact root.

## Normal evaluation

```text
results/
  run_index.csv
  <operator_id>/<candidate_id>/
    history.csv
    latest.json
    <evaluation_id>/
      evaluation_manifest.json
      results.csv
      correctness_outputs.csv
      performance_samples.csv
      model_projection.csv
      summary.md
      diagnostics/
      logs/{controller.jsonl,worker.jsonl,stdout.log,stderr.log}
```

The manifest stores identity, lifecycle, original command, resolved config,
source hashes, suite/mode, and environment snapshot/fingerprint. Legal states:

```text
planned -> running -> complete | failed | interrupted
interrupted -> running
```

Resume resolves `run_index.csv`, then requires matching path, schema, identity,
sources, config, suite/mode, and environment. Completed result IDs are skipped;
failed/complete evaluations are terminal. Only complete normal evaluations are
published to history/latest.

Controller-owned text/JSON/CSV writes use flush, fsync, and `os.replace`.
Workers write bounded attempt responses, diagnostics, and event streams; a
stale response cannot satisfy a later attempt.

## Legacy import

Legacy data uses the same mirrored directory components but a different,
strict lifecycle boundary:

```text
results/<operator_id>/<candidate_id>/<evaluation_id>/
  legacy_import_manifest.json
  results.csv
  correctness_outputs.csv       # header only
  performance_samples.csv       # header only
  model_projection.csv
  summary.md
  diagnostics/
```

`legacy_import_manifest.json` records `imported_legacy=true`, deterministic
identity, source CSV content hash, unavailable capability flags, and SHA-256
for every generated artifact. Implementation source hashes remain null.

The converter validates the complete input and stages every directory before
an atomic directory rename. A failure cannot expose a half-written evaluation.
Across a multi-operator import, earlier complete directory renames may remain;
rerunning validates and reuses them, then converges by publishing the missing
directories. Same-content reimport is idempotent; different artifact/source
facts at the deterministic target are a conflict.

Legacy directories are intentionally absent from run index/history/latest and
have no normal evaluation manifest. They are not resumable or automatically
discoverable by run summarize/compare. Direct summarize labels them; direct
compare rejects them.

Exact columns and row rules are in [CSV schemas](csv-schema.md).
