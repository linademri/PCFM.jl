#!/bin/bash
# =============================================================================
# PCFM.jl — end-to-end sample_pcfm benchmark on Engaging GPU.
#
# Trains an FFM first (saved to checkpoint), then runs benchmark_sample_pcfm.jl
# across backends and batch sizes. This is the benchmark that answers "does the
# projection backend matter for the full PCFM loop?"
#
# Submit: sbatch benchmarks/slurm/sample_pcfm_gpu.sh
# =============================================================================
#SBATCH --job-name=pcfm-e2e-gpu
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_logs/e2e_gpu_%j.out
#SBATCH --error=slurm_logs/e2e_gpu_%j.err

set -euo pipefail
mkdir -p slurm_logs results checkpoints

module load julia/1.12.6 || module load julia/1.12.6
nvidia-smi

export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd "$SLURM_SUBMIT_DIR"

julia --project=. -e 'using Pkg; Pkg.instantiate()'

# ----- Step 1: train and save a checkpoint (if we don't already have one) ----
CKPT="checkpoints/ffm_nx100_nt100.jls"
if [ ! -f "$CKPT" ]; then
    echo "=== Training FFM (no existing checkpoint) ==="
    export PCFM_BENCH_NX=100
    export PCFM_BENCH_NT=100
    export PCFM_BENCH_EPOCHS=500
    export PCFM_BENCH_TRAIN_BS=32
    julia --project=. benchmarks/train_and_save.jl "$CKPT"
else
    echo "=== Found existing checkpoint: $CKPT — skipping training ==="
fi

# ----- Step 2: end-to-end benchmark -----------------------------------------
echo "=== End-to-end sample_pcfm benchmark ==="
export PCFM_BENCH_BATCHES="16,32,64,128"
export PCFM_BENCH_STEPS=100
export PCFM_BENCH_CONSTRAINT=energy
export PCFM_BENCH_BACKENDS="gn"
export PCFM_BENCH_OUT="results/sample_pcfm_gpu_${SLURM_JOB_ID}.csv"
export PCFM_BENCH_REPEAT=3
export PCFM_BENCH_WARMUP=1

julia --project=. benchmarks/benchmark_sample_pcfm.jl "$CKPT"

echo "All done."
