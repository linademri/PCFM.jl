#!/bin/bash
#SBATCH --job-name=pcfm-reactant-smoke

#SBATCH --partition=mit_normal

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=8

#SBATCH --mem=32G

#SBATCH --time=00:30:00

#SBATCH --output=logs/pcfm-reactant-smoke-%j.out

#SBATCH --error=logs/pcfm-reactant-smoke-%j.err



set -euo pipefail



cd "$SLURM_SUBMIT_DIR"
JULIA="/orcd/software/community/001/pkg/julia/1.10.4/bin/julia"



export PATH="$HOME/software/julia-1.10.10/bin:$PATH"

export JULIA_DEPOT_PATH="$HOME/.julia-1.10"

export JULIA_PKG_PRECOMPILE_AUTO=0

export JULIA_NUM_THREADS="${SLURM_CPUS_PER_TASK}"



# Limit XLA/TensorFlow-style thread pools.

export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"

export TF_NUM_INTRAOP_THREADS=1

export TF_NUM_INTEROP_THREADS=1

export OMP_NUM_THREADS=1

export OPENBLAS_NUM_THREADS=1

export MKL_NUM_THREADS=1



mkdir -p logs



echo "Job ID: $SLURM_JOB_ID"

echo "Node: $(hostname)"

echo "Julia: $(julia --version)"

echo "Threads: $JULIA_NUM_THREADS"



"$JULIA" --project=. -e 'import Reactant; println("Reactant OK")'

"$JULIA" --project=. -e 'using PCFM; println("PCFM OK")'



echo "Finished: $(date)"

