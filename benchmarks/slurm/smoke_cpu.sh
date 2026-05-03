#!/bin/bash
# =============================================================================
# PCFM.jl — CPU smoke-test benchmark on Engaging.
#
# Purpose: validate that the package loads and the pure-Julia backends run before
# burning GPU time on the full benchmarks. This is what you run FIRST after cloning
# on Engaging, to catch environment issues early.
#
# Runs projection-only benchmark on Gauss-Newton and MadNLP (CPU) backends, small
# problem size, small batch sweep. ~5-10 minutes.
#
# Submit: sbatch benchmarks/slurm/smoke_cpu.sh
# =============================================================================
#SBATCH --job-name=pcfm-smoke
#SBATCH --partition=mit_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=slurm_logs/smoke_%j.out
#SBATCH --error=slurm_logs/smoke_%j.err

# ----- Environment setup ----------------------------------------------------
set -euo pipefail
mkdir -p slurm_logs

# Load Julia — adjust module name to what Engaging has (check `module avail julia`).
module load julia/1.10.4 || module load julia/1.10.4
module list

# Reactant/CUDA-free run — force CPU backend to avoid surprises.
export JULIA_CUDA_USE_BINARYBUILDER=false

# Keep BLAS single-threaded; we're using Slurm's CPU allocation directly.
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd "$SLURM_SUBMIT_DIR"

# First-time package instantiation. On subsequent runs this is a no-op.
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# ----- Benchmark 1: projection-only, tiny -----------------------------------
echo "=== Projection benchmark (CPU, small) ==="
export PCFM_BENCH_BATCHES="16,32,64"
export PCFM_BENCH_NX=32
export PCFM_BENCH_NT=32
export PCFM_BENCH_CONSTRAINT=energy
export PCFM_BENCH_BACKENDS="gn,madnlp"
export PCFM_BENCH_OUT="results/smoke_projection_${SLURM_JOB_ID}.csv"
export PCFM_BENCH_REPEAT=3
export PCFM_BENCH_WARMUP=1
mkdir -p results

julia --project=. benchmarks/benchmark_projection.jl

echo "Done. Results in $PCFM_BENCH_OUT"
