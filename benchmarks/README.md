# PCFM.jl Benchmarks on MIT Engaging

Self-contained benchmarking pipeline for the PCFM v0.2 extension, designed for MIT ORCD's
Engaging cluster. Answers the four benchmark questions from the 18.337 project proposal:

1. **Constraint violation `‖h(z)‖`** — correctness check across backends.
2. **Projection wall time vs. batch size** — isolated from the ODE solve. → `benchmark_projection.jl`
3. **End-to-end `sample_pcfm` wall time** — projection cost in context. → `benchmark_sample_pcfm.jl`
4. **MadNLP KKT structure exploitation** — block-diagonal vs. unstructured NLP formulation. → `benchmark_madnlp_sparsity.jl`

## First-time setup on Engaging

```bash
# Log in (via OnDemand portal first time to activate your account)
ssh <kerberos>@eofe10.mit.edu

# Clone and set up
git clone <your fork of PCFM.jl>
cd PCFM-v0.2

# Figure out which Julia module the cluster has
module avail julia
module load julia/1.10   # or whatever you found

# Instantiate dependencies (first time takes ~10 minutes for Reactant + deps)
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# Run unit tests on a login node (or interactive session) to confirm it works
julia --project=. -e 'using Pkg; Pkg.test()'
```

If `Pkg.test()` fails, fix that before running benchmarks — the most common failures are
(a) missing CUDA libraries for Reactant's jll artifacts, (b) MadNLP extension load errors,
(c) Julia version mismatch. See `../docs/projection_pr.md` for likely-first-bug locations.

## Running the benchmarks

All benchmarks are driven by `sbatch` scripts under `slurm/`. Submit them from the repo
root (so relative paths resolve correctly):

```bash
# Recommended order:

# 1. 30-min CPU smoke test — validates the stack before burning GPU time.
sbatch benchmarks/slurm/smoke_cpu.sh

# 2. Main scaling plot — projection-only, GPU, full batch sweep (~2 h).
sbatch benchmarks/slurm/projection_gpu.sh

# 3. MadNLP structural comparison — CPU, no GPU needed (~1.5 h).
sbatch benchmarks/slurm/madnlp_sparsity.sh

# 4. End-to-end — trains FFM then benchmarks sample_pcfm (~6 h, includes training).
sbatch benchmarks/slurm/sample_pcfm_gpu.sh
```

Check status:

```bash
squeue -u $USER
tail -f slurm_logs/*_$JOBID.out
```

When jobs complete, CSVs land in `results/`. Turn them into PNGs:

```bash
julia --project=. benchmarks/plot_results.jl results/*.csv
```

## Configuration

All benchmarks accept env vars so you don't need to edit code for parameter sweeps:

| Env var                       | Default     | Meaning                                         |
|-------------------------------|-------------|-------------------------------------------------|
| `PCFM_BENCH_BATCHES`          | `32,64,...` | Comma-separated batch sizes                     |
| `PCFM_BENCH_NX`               | `100`       | Spatial resolution                              |
| `PCFM_BENCH_NT`               | `100`       | Temporal resolution                             |
| `PCFM_BENCH_CONSTRAINT`       | `energy`    | `linear_ic` \| `mass` \| `energy`               |
| `PCFM_BENCH_BACKENDS`         | `gn`        | Comma of: `gn`, `madnlp`, `madnlp_gpu`, `optimjl` |
| `PCFM_BENCH_STEPS`            | `100`       | Euler steps (end-to-end only)                   |
| `PCFM_BENCH_WARMUP`           | `3`         | Warmup iters                                    |
| `PCFM_BENCH_REPEAT`           | `10`        | Timed iters                                     |
| `PCFM_BENCH_OUT`              | `*.csv`     | Output CSV path                                 |
| `PCFM_BENCH_SEED`             | `20260420`  | RNG seed                                        |

## Partition notes

Engaging uses the Slurm scheduler. Relevant partitions:

- `mit_normal` — CPU, public, no time limit quirks; use for MadNLP/CPU benchmarks.
- `mit_normal_gpu` — GPU, public; starting point for GPU benchmarks.
- `mit_preemptable` — has H100/H200/L40S but jobs can be preempted. Fine for benchmarks
  that save intermediate state; pass `--constraint=a100` (or `h100`, `l40s`) to pick a type.

If you're using CSAIL TIG's Slurm cluster instead of Engaging, change the header to:

```
#SBATCH --account=csail
#SBATCH --partition=tig-gpu
#SBATCH --qos=tig-main
```

and submit from `slurm-login.csail.mit.edu`.

## Output format

All benchmark scripts write long-format CSVs. Columns:

- `benchmark_projection.jl` → `backend, batch_size, constraint, wall_time_s_*, max_violation, cold_*`
- `benchmark_sample_pcfm.jl` → `backend, batch_size, wall_time_s_*, per_step_ms`
- `benchmark_madnlp_sparsity.jl` → `batch_size, variant, build_s, solve_s, iters, status`

This makes it easy to pull into pandas/Julia DataFrames or push to a notebook for further
analysis without touching the benchmark code.

## Troubleshooting

**"Invalid account or account/partition combination"** — wait 15 minutes after account
activation; see `https://orcd-docs.mit.edu/faqs/`.

**Reactant fails to precompile with `libReactantExtra.so`** — usually a CUDA version
mismatch. PCFM uses Reactant for the FFM model itself (not for any projection backend),
so this affects training and sampling regardless of which projection solver you pick. On
Engaging the fix is typically to `module load cuda/12.2` (or whatever matches the
Reactant_jll artifact version) before starting Julia.

**MadNLP tests skip silently** — `using ExaModels, MadNLP` failed inside the test harness.
Run `julia --project=. -e 'using ExaModels, MadNLP'` interactively to see the actual error.

## What to report

For an 18.337 writeup, the headline numbers to pull out of the CSVs:

1. **Scaling plot** from `benchmark_projection.jl`: wall time vs. batch size for each
   backend on a log-log plot. Slope tells you the asymptotic scaling; intercept tells you
   the per-call overhead.
2. **Constraint violation table** from the same run — confirms every backend lands below
   its declared tolerance. This is the correctness anchor.
3. **Structured vs. unstructured MadNLP** bar chart from `benchmark_madnlp_sparsity.jl` —
   the specific "does ExaModels' SIMD abstraction matter?" data point.
4. **Projection share of end-to-end time** from `benchmark_sample_pcfm.jl`: for each
   backend, `projection_time / total_time`. This is the "does the projection backend
   matter for real PCFM workloads?" number.
