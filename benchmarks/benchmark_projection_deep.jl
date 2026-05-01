#= 
Projection benchmark with evaluation-depth outputs.

Adds, relative to benchmark_projection.jl:
  - manifest TOML for reproducibility
  - mean/median/p95/p99/max violation, not only max
  - failure counts and failure rate
  - projection displacement statistics
  - cold and warm timing separation

Run:
  julia --project=. benchmarks/benchmark_projection_deep.jl
=#

using PCFM
using Random
using Statistics
using LinearAlgebra
using Printf

include(joinpath(@__DIR__, "BenchmarkUtils.jl"))
using .BenchmarkUtils

const BATCHES = parse_int_list(get(ENV, "PCFM_BENCH_BATCHES", "32,64,128,256"))
const NX = env_int("PCFM_BENCH_NX", 100)
const NT = env_int("PCFM_BENCH_NT", 100)
const CONSTRAINT = Symbol(env_string("PCFM_BENCH_CONSTRAINT", "energy"))
const BACKENDS = parse_symbol_list(get(ENV, "PCFM_BENCH_BACKENDS", "gn"))
const OUT_PATH = env_string("PCFM_BENCH_OUT", "results/projection_deep.csv")
const WARMUP = env_int("PCFM_BENCH_WARMUP", 3)
const REPEAT = env_int("PCFM_BENCH_REPEAT", 10)
const SEED = env_int("PCFM_BENCH_SEED", 20260420)
const TOL = env_float("PCFM_BENCH_TOL", 1e-6)

Random.seed!(SEED)

const HEADER = [
    "benchmark", "backend", "batch_size", "constraint", "nx", "nt", "seed", "warmup", "repeat",
    "cold_time_s", "cold_max_violation",
    "wall_time_s_mean", "wall_time_s_std", "wall_time_s_min", "wall_time_s_median",
    "mean_violation", "median_violation", "p95_violation", "p99_violation", "max_violation",
    "n_failed", "failure_rate",
    "projection_displacement_mean", "projection_displacement_p95", "projection_displacement_max",
    "status",
]

function displacement_stats(Z, Zhat)
    nx, nt, _, nb = size(Z)
    Zf = reshape(Float64.(Array(Z)), nx * nt, nb)
    Hf = reshape(Float64.(Array(Zhat)), nx * nt, nb)
    d = [norm(Zf[:, i] - Hf[:, i]) for i in 1:nb]
    return Dict(
        "projection_displacement_mean" => mean(d),
        "projection_displacement_p95" => quantile(d, 0.95),
        "projection_displacement_max" => maximum(d),
    )
end

function time_one(backend::Symbol, nb::Int, constraint)
    solver, status = make_solver(backend)
    if solver === nothing
        row = Dict{String, Any}(
            "benchmark" => "projection_deep", "backend" => backend, "batch_size" => nb,
            "constraint" => CONSTRAINT, "nx" => NX, "nt" => NT, "seed" => SEED,
            "warmup" => WARMUP, "repeat" => REPEAT, "status" => sprint(showerror, status),
        )
        append_csv_row(OUT_PATH, HEADER, row)
        @warn "Backend skipped" backend exception=status
        return
    end

    Zhat = randn(Float32, NX, NT, 1, nb)

    t_cold = @elapsed Z = PCFM.project(solver, Zhat, constraint)
    cold_stats = violation_stats(Z, constraint; tol = TOL)

    for _ in 1:WARMUP
        PCFM.project(solver, Zhat, constraint)
    end

    times = Float64[]
    Z_last = Z
    for _ in 1:REPEAT
        t = @elapsed Z_last = PCFM.project(solver, Zhat, constraint)
        push!(times, t)
    end

    vstats = violation_stats(Z_last, constraint; tol = TOL)
    dstats = displacement_stats(Z_last, Zhat)

    row = Dict{String, Any}(
        "benchmark" => "projection_deep",
        "backend" => backend,
        "batch_size" => nb,
        "constraint" => CONSTRAINT,
        "nx" => NX,
        "nt" => NT,
        "seed" => SEED,
        "warmup" => WARMUP,
        "repeat" => REPEAT,
        "cold_time_s" => t_cold,
        "cold_max_violation" => cold_stats["max_violation"],
        "wall_time_s_mean" => mean(times),
        "wall_time_s_std" => length(times) > 1 ? std(times) : 0.0,
        "wall_time_s_min" => minimum(times),
        "wall_time_s_median" => median(times),
        "status" => "ok",
    )
    merge!(row, vstats)
    merge!(row, dstats)
    append_csv_row(OUT_PATH, HEADER, row)

    @info "projection_deep" backend nb wall_time_s_mean=row["wall_time_s_mean"] max_violation=row["max_violation"] failure_rate=row["failure_rate"]
end

manifest_path = write_manifest(
    OUT_PATH;
    benchmark = "projection_deep",
    config = Dict(
        "batches" => BATCHES,
        "nx" => NX,
        "nt" => NT,
        "constraint" => string(CONSTRAINT),
        "backends" => string.(BACKENDS),
        "warmup" => WARMUP,
        "repeat" => REPEAT,
        "seed" => SEED,
        "tol" => TOL,
    ),
)
@info "wrote manifest" manifest_path

constraint = make_constraint(CONSTRAINT, NX, NT)
for backend in BACKENDS, nb in BATCHES
    time_one(backend, nb, constraint)
end

@info "wrote results" OUT_PATH
