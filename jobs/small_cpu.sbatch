#!/bin/bash

#SBATCH --job-name=pcfm-small-cpu

#SBATCH --partition=mit_normal

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=8

#SBATCH --mem=32G

#SBATCH --time=02:00:00

#SBATCH --output=logs/pcfm-small-cpu-%j.out

#SBATCH --error=logs/pcfm-small-cpu-%j.err



set -euo pipefail



cd "$SLURM_SUBMIT_DIR"
JULIA="/orcd/software/community/001/pkg/julia/1.10.4/bin/julia"



export JULIA_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

export JULIA_PKG_PRECOMPILE_AUTO=0



mkdir -p logs results



echo "Job ID: $SLURM_JOB_ID"

echo "Node: $(hostname)"

echo "Started: $(date)"

echo "Commit: $(git rev-parse --short HEAD || true)"

"$JULIA" --version



echo "Skipping Pkg.precompile because it fails with Enzyme KeyError on this environment"



echo "Checking imports"

"$JULIA" --project=. -e 'import MadNLP, ExaModels, Optimization, OptimizationOptimJL; using PCFM; println("Imports OK")'



echo "Running small CPU benchmark"

"$JULIA" --project=. benchmarks/run_suite.jl benchmarks/suites/small_cpu.toml



echo "Generating report"

"$JULIA" --project=benchmarks benchmarks/make_report.jl results/small_cpu/*.csv || true



echo "Finished: $(date)"

