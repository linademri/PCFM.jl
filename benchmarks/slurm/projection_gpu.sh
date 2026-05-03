#!/bin/bash
# =============================================================================
# PCFM.jl — full projection benchmark on Engaging GPU.
#
# Runs benchmark_projection.jl across all batch sizes with the CPU-array Gauss-Newton
# backend on a GPU node (the backend uses whatever array type PCFM's `device` provides,
# so it runs on GPU when device = reactant_device()). Produces the main scaling plot data.
#
# Submit: sbatch benchmarks/slurm/projection_gpu.sh
# =============================================================================
#SBATCH --job-name=pcfm-proj-gpu
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/proj_gpu_%j.out
#SBATCH --error=slurm_logs/proj_gpu_%j.err

# Optional: uncomment to target a specific GPU type. `mit_preemptable` has more variety
# (H100/H200/A100/L40S) but jobs are preemptable — fine for benchmarks that can restart.
# #SBATCH --partition=mit_preemptable
# #SBATCH --constraint=a100    # or h100, l40s, rtx6000

set -euo pipefail
mkdir -p slurm_logs results

module load julia/1.10.4 || module load julia/1.10.4
module load cuda/12.2 || module load cuda || true    # Reactant brings its own CUDA via jll
nvidia-smi   # log GPU for the run

export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd "$SLURM_SUBMIT_DIR"

julia --project=. -e 'using Pkg; Pkg.instantiate()'

# ----- Benchmark: projection-only, full sweep ---------------------------------
export PCFM_BENCH_BATCHES="32,64,128,256,512,1024"
export PCFM_BENCH_NX=100
export PCFM_BENCH_NT=100
export PCFM_BENCH_CONSTRAINT=energy
export PCFM_BENCH_BACKENDS="gn,madnlp,madnlp_gpu"
export PCFM_BENCH_OUT="results/projection_gpu_${SLURM_JOB_ID}.csv"
export PCFM_BENCH_REPEAT=10
export PCFM_BENCH_WARMUP=3

julia --project=. benchmarks/benchmark_projection.jl

# Also run with LinearIC and mass constraints for a comparison across constraint difficulty.
for cstr in linear_ic mass; do
    echo "=== Constraint: $cstr ==="
    export PCFM_BENCH_CONSTRAINT=$cstr
    export PCFM_BENCH_OUT="results/projection_gpu_${cstr}_${SLURM_JOB_ID}.csv"
    julia --project=. benchmarks/benchmark_projection.jl
done

echo "All done. CSVs under results/."
