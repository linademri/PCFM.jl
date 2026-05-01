# Applying the evaluation-depth patch

Copy the contents of this directory into the root of `linademri/PCFM.jl`.

```bash
rsync -av pcfm_eval_depth_patch/ /path/to/PCFM.jl/
cd /path/to/PCFM.jl
julia --project=. benchmarks/run_suite.jl benchmarks/suites/small_cpu.toml
julia --project=benchmarks benchmarks/make_report.jl results/small_cpu/*.csv
```

The new files are additive. They do not remove or replace the existing benchmark scripts.

## Added files

- `benchmarks/BenchmarkUtils.jl`
- `benchmarks/benchmark_projection_deep.jl`
- `benchmarks/benchmark_sample_quality.jl`
- `benchmarks/run_suite.jl`
- `benchmarks/make_report.jl`
- `benchmarks/suites/small_cpu.toml`
- `benchmarks/suites/main_gpu.toml`
- `benchmarks/suites/stress.toml`
- `docs/evaluation.md`

## Notes

I could not execute Julia validation in the sandbox because Julia is not installed there. The scripts avoid non-stdlib dependencies except `make_report.jl`, which intentionally uses the existing `benchmarks/Project.toml` packages.
