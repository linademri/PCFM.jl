#!/bin/bash
#SBATCH --job-name=pcfm-main-gpu
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/pcfm-main-gpu-%j.out
#SBATCH --error=logs/pcfm-main-gpu-%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
JULIA="/orcd/software/community/001/pkg/julia/1.10.4/bin/julia"

module purge || true

export JULIA_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export JULIA_PKG_PRECOMPILE_AUTO=0
export PCFM_BENCH_OUTDIR="results/main_gpu_${SLURM_JOB_ID}"

mkdir -p logs "$PCFM_BENCH_OUTDIR"

echo "======================================================================"
echo "PCFM main GPU benchmark"
echo "======================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "Repo: $(pwd)"
echo "Commit: $(git rev-parse --short HEAD || true)"
"$JULIA" --version

echo
echo "nvidia-smi:"
nvidia-smi

echo
echo "Checking CUDA..."
"$JULIA" --project=. -e '
using CUDA
CUDA.versioninfo()
@assert CUDA.functional()
println("CUDA OK")
'

echo

echo "Checking MadNLPGPU extension..."

"$JULIA" --project=. -e '

using CUDA

using ExaModels

using MadNLP

using MadNLPGPU

using PCFM

println("MadNLPGPU extension imports OK")

'



echo
echo "Running main GPU suite..."
"$JULIA" --project=. benchmarks/run_suite.jl benchmarks/suites/main_gpu.toml

echo
echo "Generating report..."
"$JULIA" --project=benchmarks benchmarks/make_report.jl "$PCFM_BENCH_OUTDIR"/*.csv || true

echo
echo "Finished: $(date)"
