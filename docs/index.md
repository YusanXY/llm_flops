# Benchmark engine documentation

Start with [getting started](getting-started.md) and the [CLI reference](cli.md).
The root [README](../README.md) contains the shortest runnable example.

## Use the engine

- [KDA-Pilot integration](kda-pilot-integration.md): task-native baseline,
  versioned solutions, one-command benchmark, runtime isolation, and `bench/`
  artifacts. Agents should read this first when working inside KDA-Pilot.
- [Getting started](getting-started.md): install, discover, dry-run, execute, resume.
- [CLI](cli.md): commands, selectors, precedence, and exit codes.
- [Troubleshooting](troubleshooting.md): JIT, timeout, OOM, locks, noise, and artifacts.
- [Test tiers](testing.md): CPU, GPU smoke, B200 regression, and nightly/full gates.

## Add an implementation

- [Operator contract](operator-contract.md): `operator.yaml`, `spec.py`, reference entrypoint.
- [Candidate guide](candidate-guide.md): layout, naming, metadata, and source identity.
- [Correctness](correctness.md): isolated inputs, observed state, comparators, diagnostics.
- [Performance](performance.md): staged timers, samples, fairness, gates, and cost models.
- [Minimal operator](examples/minimal-operator/operator.yaml) and
  [minimal candidate](examples/minimal-candidate/implementation.py).

## Understand artifacts and migration

- [Architecture](architecture.md) and [implementation guide](implementation.md).
- [Result layout](result-layout.md) and [CSV schema](csv-schema.md).
- [Legacy llm_flops migration](migration-llm-flops.md).
- [GLM-5 migration and coverage](glm5-migration.md).
- [Legacy launchers](legacy-launchers.md).
- [Development](development.md) and [contributing](../CONTRIBUTING.md).

Operator-specific shapes, layouts, tolerances, cost models, and commands live in
`baseline/README.md` for KDA tasks and
`operators/references/<operator_id>/README.md` in repository mode.
