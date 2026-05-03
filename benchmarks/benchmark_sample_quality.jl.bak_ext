#=
Sample-quality benchmark for constrained projection backends.

This is a synthetic distribution-level check for projection quality. It compares each
backend's projected samples against a reference projection of the same unconstrained
inputs. By default the reference backend is Gauss-Newton (`PCFM_BENCH_REFERENCE_BACKEND=gn`).

For full paper-style evaluation, replace the synthetic Zhat generator with model samples
and replace Zref with held-out ground-truth trajectories; the output schema remains the same.

Run:
  julia --project=. benchmarks/benchmark_sample_quality.jl
=#

using PCFM
using Random
using Statistics
using LinearAlgebra

include(joinpath(@__DIR__, "BenchmarkUtils.jl"))
using .BenchmarkUtils

const BATCHES = parse_int_list(get(ENV, "PCFM_BENCH_BATCHES", "32,64,128"))
const NX = env_int("PCFM_BENCH_NX", 100)
const NT = env_int("PCFM_BENCH_NT", 100)
const CONSTRAINT = Symbol(env_string("PCFM_BENCH_CONSTRAINT", "energy"))
const BACKENDS = parse_symbol_list(get(ENV, "PCFM_BENCH_BACKENDS", "gn,madnlp,optimjl"))
const REFERENCE_BACKEND = Symbol(env_string("PCFM_BENCH_REFERENCE_BACKEND", "gn"))
const OUT_PATH = env_string("PCFM_BENCH_OUT", "results/sample_quality.csv")
const SEED = env_int("PCFM_BENCH_SEED", 20260420)
const TOL = env_float("PCFM_BENCH_TOL", 1e-6)

Random.seed!(SEED)

const HEADER = [
    "benchmark", "backend", "reference_backend", "batch_size", "constraint", "nx", "nt", "seed",
    "wall_time_s", "reference_time_s",
    "mmd_rbf", "energy_distance", "mean_l2", "cov_frobenius", "spectral_l2",
    "mean_violation", "median_violation", "p95_violation", "p99_violation", "max_violation",
    "n_failed", "failure_rate", "projection_displacement_mean", "status",
]

function displacement_mean(Z, Zhat)
    nx, nt, _, nb = size(Z)
    Zf = reshape(Float64.(Array(Z)), nx * nt, nb)
    Hf = reshape(Float64.(Array(Zhat)), nx * nt, nb)
    return mean(norm(Zf[:, i] - Hf[:, i]) for i in 1:nb)
end

function evaluate_one(backend::Symbol, nb::Int, constraint, Zhat, Zref, reference_time)
    solver, status = make_solver(backend)
    if solver === nothing
        append_csv_row(OUT_PATH, HEADER, Dict{String, Any}(
            "benchmark" => "sample_quality", "backend" => backend,
            "reference_backend" => REFERENCE_BACKEND, "batch_size" => nb,
            "constraint" => CONSTRAINT, "nx" => NX, "nt" => NT, "seed" => SEED,
            "status" => sprint(showerror, status),
        ))
        @warn "Backend skipped" backend exception=status
        return
    end

    t = @elapsed Z = PCFM.project(solver, Zhat, constraint)
    quality = distribution_metrics(Z, Zref)
    vstats = violation_stats(Z, constraint; tol = TOL)

    row = Dict{String, Any}(
        "benchmark" => "sample_quality",
        "backend" => backend,
        "reference_backend" => REFERENCE_BACKEND,
        "batch_size" => nb,
        "constraint" => CONSTRAINT,
        "nx" => NX,
        "nt" => NT,
        "seed" => SEED,
        "wall_time_s" => t,
        "reference_time_s" => reference_time,
        "projection_displacement_mean" => displacement_mean(Z, Zhat),
        "status" => "ok",
    )
    merge!(row, quality)
    merge!(row, vstats)
    append_csv_row(OUT_PATH, HEADER, row)

    @info "sample_quality" backend nb mmd=row["mmd_rbf"] max_violation=row["max_violation"]
end

manifest_path = write_manifest(
    OUT_PATH;
    benchmark = "sample_quality",
    config = Dict(
        "batches" => BATCHES,
        "nx" => NX,
        "nt" => NT,
        "constraint" => string(CONSTRAINT),
        "backends" => string.(BACKENDS),
        "reference_backend" => string(REFERENCE_BACKEND),
        "seed" => SEED,
        "tol" => TOL,
    ),
)
@info "wrote manifest" manifest_path

constraint = make_constraint(CONSTRAINT, NX, NT)
ref_solver, ref_status = make_solver(REFERENCE_BACKEND)
ref_solver === nothing && error("Reference backend $REFERENCE_BACKEND is unavailable: $ref_status")

for nb in BATCHES
    Zhat = randn(Float32, NX, NT, 1, nb)
    tref = @elapsed Zref = PCFM.project(ref_solver, Zhat, constraint)
    for backend in BACKENDS
        evaluate_one(backend, nb, constraint, Zhat, Zref, tref)
    end
end

@info "wrote results" OUT_PATH
