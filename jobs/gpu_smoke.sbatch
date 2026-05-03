#!/bin/bash
#SBATCH --job-name=pcfm-gpu-smoke
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/pcfm-gpu-smoke-%j.out
#SBATCH --error=logs/pcfm-gpu-smoke-%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
JULIA="/orcd/software/community/001/pkg/julia/1.10.4/bin/julia"
echo "PATH=$PATH"
echo "Julia path: $(which julia)"
"$JULIA" --version



echo

echo "nvidia-smi:"

nvidia-smi



echo

echo "Checking CUDA..."

"$JULIA" --project=. --compiled-modules=no -e '

using CUDA

CUDA.versioninfo()

@assert CUDA.functional()

println("CUDA OK")

'



echo

echo "Checking MadNLPGPU without loading PCFM..."

"$JULIA" --project=. --compiled-modules=no -e '

using CUDA

using MadNLP

using ExaModels

using MadNLPGPU

println("MadNLPGPU OK")

'



echo

echo "Finished: $(date)"
