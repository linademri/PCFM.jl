#!/bin/bash
# =============================================================================
# PCFM.jl — MadNLP KKT / sparsity-structure benchmark on Engaging CPU.
#
# Answers the proposal's specific question: does ExaModels' block-diagonal SIMD
# abstraction meaningfully reduce solve time compared to a formulation that hides
# the per-sample structure? This is pure algebra/assembly cost — no GPU needed.
#
# Submit: sbatch benchmarks/slurm/madnlp_sparsity.sh
# =============================================================================
#SBATCH --job-name=pcfm-madnlp-sparsity
#SBATCH --partition=mit_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=slurm_logs/madnlp_sparsity_%j.out
#SBATCH --error=slurm_logs/madnlp_sparsity_%j.err

set -euo pipefail
mkdir -p slurm_logs results

module load julia/1.10.4 || module load julia/1.10.4

export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd "$SLURM_SUBMIT_DIR"

julia --project=. -e 'using Pkg; Pkg.instantiate()'

# Sweep batch sizes on a smaller nx/nt to keep per-call times reasonable — this is a
# structural diagnostic, not a scaling plot.
export PCFM_BENCH_BATCHES="64,128,256,512,1024"
export PCFM_BENCH_NX=32
export PCFM_BENCH_NT=32
export PCFM_BENCH_CONSTRAINT=energy
export PCFM_BENCH_OUT="results/madnlp_sparsity_${SLURM_JOB_ID}.csv"

julia --project=. benchmarks/benchmark_madnlp_sparsity.jl

echo "Done. Results in $PCFM_BENCH_OUT"
