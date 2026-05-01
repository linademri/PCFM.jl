# Evaluation depth plan

This repo now has an evaluation layer that separates implementation correctness from experimental evidence.

## Scripts

- `benchmarks/benchmark_projection_deep.jl` records cold/warm projection time, violation distributions, failure rate, and projection displacement.
- `benchmarks/benchmark_sample_quality.jl` records distribution-level diagnostics: MMD, energy distance, mean error, covariance error, and terminal-slice spectral error.
- `benchmarks/run_suite.jl` runs named TOML suites from `benchmarks/suites/`.
- `benchmarks/make_report.jl` converts CSV outputs into this report format.

## Recommended commands

```bash
julia --project=. benchmarks/run_suite.jl benchmarks/suites/small_cpu.toml
julia --project=benchmarks benchmarks/make_report.jl results/small_cpu/*.csv
```

For GPU runs:

```bash
julia --project=. benchmarks/run_suite.jl benchmarks/suites/main_gpu.toml
julia --project=benchmarks benchmarks/make_report.jl results/main_gpu/*.csv
```

## Output schema

Every deep benchmark writes a CSV and a sibling `*.manifest.toml` file. The manifest records the run configuration, Julia version, git commit, CPU/GPU-visible environment, Slurm metadata, and `PCFM_BENCH_*` variables.

## Interpreting results

Use this order:

1. Confirm backend status is `ok`.
2. Check `failure_rate`, `p95_violation`, and `max_violation`.
3. Compare warm timing only after correctness passes.
4. Use sample-quality metrics to detect whether projection preserves the target distribution.
5. Use cold timing and manifest metadata to explain compile/build costs.

## Notes

`benchmark_sample_quality.jl` defaults to a synthetic reference distribution produced by a reference projection backend. For paper-quality results, keep the same CSV schema but replace the synthetic `Zhat`/`Zref` construction with trained model samples and held-out PDE trajectories.
