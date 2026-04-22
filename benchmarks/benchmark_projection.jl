#=
benchmark_projection.jl

Projection-only benchmark. Isolates the cost of the PCFM projection step from the rest of
the sampling loop by feeding synthetic `Ẑ` batches of varying size to each solver backend
and measuring (a) wall time, (b) maximum constraint violation.

This is the most informative benchmark because it answers the question the proposal poses
directly — "Inference wall time vs. batch size, isolating the cost of the projection step
relative to the ODE solve". By removing the FNO entirely, we can attribute every
microsecond to the projection.

Usage:

    julia --project=. benchmarks/benchmark_projection.jl

Or under sbatch — see benchmarks/slurm/*.sh.

Environment variables:
    PCFM_BENCH_BATCHES="32,64,128,256,512,1024"   # comma-separated batch sizes
    PCFM_BENCH_NX=100                              # spatial resolution
    PCFM_BENCH_NT=100                              # temporal resolution
    PCFM_BENCH_CONSTRAINT=energy                   # linear_ic | mass | energy
    PCFM_BENCH_BACKENDS="gn,madnlp"                # which solvers to test
    PCFM_BENCH_OUT=results.csv                     # output path
    PCFM_BENCH_WARMUP=3                            # warmup iters before timing
    PCFM_BENCH_REPEAT=10                           # timed iters to average

The CSV output has columns: backend, batch_size, constraint, wall_time_s, max_violation,
compile_time_s (MadNLP build time; 0 for other backends), n_samples_satisfying.
=#

using PCFM
using Random
using LinearAlgebra
using Statistics: mean, std
using Printf

# -----------------------------------------------------------------------------
# Configuration from environment
# -----------------------------------------------------------------------------
const BATCHES     = parse.(Int, split(get(ENV, "PCFM_BENCH_BATCHES", "32,64,128,256"), ","))
const NX          = parse(Int, get(ENV, "PCFM_BENCH_NX", "100"))
const NT          = parse(Int, get(ENV, "PCFM_BENCH_NT", "100"))
const CONSTRAINT  = Symbol(get(ENV, "PCFM_BENCH_CONSTRAINT", "energy"))
const BACKENDS    = Symbol.(split(get(ENV, "PCFM_BENCH_BACKENDS", "gn"), ","))
const OUT_PATH    = get(ENV, "PCFM_BENCH_OUT", "results_projection.csv")
const WARMUP      = parse(Int, get(ENV, "PCFM_BENCH_WARMUP", "3"))
const REPEAT      = parse(Int, get(ENV, "PCFM_BENCH_REPEAT", "10"))
const SEED        = parse(Int, get(ENV, "PCFM_BENCH_SEED", "20260420"))

Random.seed!(SEED)

# -----------------------------------------------------------------------------
# Constraint factory
# -----------------------------------------------------------------------------
function make_constraint(kind::Symbol, nx::Int, nt::Int)
    if kind === :linear_ic
        x_grid = range(0, 2π, length = nx)
        u0 = Float32.(sin.(x_grid .+ π / 4))
        return LinearICConstraint(u0, nx, nt)
    elseif kind === :mass
        return MassConservationConstraint(0.0f0; nx = nx, nt = nt)
    elseif kind === :energy
        return EnergyConservationConstraint(1.0f0; nx = nx, nt = nt)
    else
        error("Unknown constraint kind: $kind")
    end
end

# -----------------------------------------------------------------------------
# Backend factory. Deferred imports so we don't require every extension to be
# installed to run partial benchmarks.
# -----------------------------------------------------------------------------
function make_solver(backend::Symbol)
    if backend === :gn
        return BatchedGaussNewtonSolver(tol = 1e-7, max_iter = 25), :always_available
    elseif backend === :madnlp
        try
            @eval using ExaModels, MadNLP
            return MadNLPSolver(tol = 1e-8, print_level = MadNLP.ERROR), :loaded
        catch e
            return nothing, e
        end
    elseif backend === :madnlp_gpu
        try
            @eval using ExaModels, MadNLP, MadNLPGPU, CUDA
            return MadNLPGPUSolver(tol = 1e-8), :loaded
        catch e
            return nothing, e
        end
    elseif backend === :optimjl
        try
            @eval using Optimization, OptimizationOptimJL
            return OptimizationJLSolver(tol = 1e-6), :loaded
        catch e
            return nothing, e
        end
    else
        error("Unknown backend: $backend")
    end
end

# -----------------------------------------------------------------------------
# Constraint-violation measurement. Computes max_i ‖h(z_i)‖∞ over the batch,
# by iterating over samples in Julia — cheap post-projection, works for any
# constraint with a `residual` method.
# -----------------------------------------------------------------------------
function max_violation(Z::AbstractArray{T, 4}, constraint) where {T}
    nx, nt, _, Nb = size(Z)
    n = nx * nt
    Z_flat = reshape(Z, n, Nb)
    worst = zero(Float64)
    for i in 1:Nb
        z_i = collect(Z_flat[:, i])  # host copy in case Z is a device array
        h = PCFM.residual(constraint, z_i)
        worst = max(worst, Float64(maximum(abs, h)))
    end
    return worst
end

# -----------------------------------------------------------------------------
# Timing one (backend, batch_size, constraint) triple.
# -----------------------------------------------------------------------------
function time_one(backend::Symbol, Nb::Int, constraint, nx::Int, nt::Int)
    solver, status = make_solver(backend)
    if solver === nothing
        @warn "Backend $backend not available — skipping" exception = status
        return nothing
    end

    # Fresh random input per (backend, Nb). Using Float32 throughout to match the FFM
    # convention. For MadNLP the solver internally promotes to Float64; that cost is part
    # of what we measure.
    Ẑ = randn(Float32, nx, nt, 1, Nb)

    # Cold call: assembles ExaModel for MadNLP; for pure-Julia backends this is just
    # a warmup. Timed separately so warm numbers aren't polluted by build cost.
    t_cold = @elapsed Z = PCFM.project(solver, Ẑ, constraint)
    viol_cold = max_violation(Z, constraint)

    # Warmup.
    for _ in 1:WARMUP
        PCFM.project(solver, Ẑ, constraint)
    end

    # Timed repeats.
    times = Float64[]
    viols = Float64[]
    for _ in 1:REPEAT
        t = @elapsed Z = PCFM.project(solver, Ẑ, constraint)
        push!(times, t)
        push!(viols, max_violation(Z, constraint))
    end

    return (
        backend = backend,
        batch_size = Nb,
        constraint = nameof(typeof(constraint)),
        wall_time_s_mean = mean(times),
        wall_time_s_std  = std(times),
        wall_time_s_min  = minimum(times),
        cold_time_s      = t_cold,
        max_violation    = maximum(viols),
        cold_violation   = viol_cold,
    )
end

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
function main()
    constraint = make_constraint(CONSTRAINT, NX, NT)

    println("=" ^ 70)
    println("PCFM projection benchmark")
    println("=" ^ 70)
    @printf "  nx = %d   nt = %d   constraint = %s\n" NX NT CONSTRAINT
    @printf "  backends: %s\n" join(BACKENDS, ", ")
    @printf "  batch sizes: %s\n" join(BATCHES, ", ")
    @printf "  warmup = %d   repeat = %d\n" WARMUP REPEAT
    println()

    results = []
    for backend in BACKENDS
        for Nb in BATCHES
            @printf "→ %-14s Nb = %4d  " backend Nb
            r = time_one(backend, Nb, constraint, NX, NT)
            if r === nothing
                println("(skipped)")
                continue
            end
            push!(results, r)
            @printf "%.4f s ± %.4f   viol = %.2e   cold = %.3f s\n" r.wall_time_s_mean r.wall_time_s_std r.max_violation r.cold_time_s
        end
    end

    # Write CSV.
    open(OUT_PATH, "w") do io
        println(io, "backend,batch_size,constraint,wall_time_s_mean,wall_time_s_std,wall_time_s_min,cold_time_s,max_violation,cold_violation")
        for r in results
            println(io, join([r.backend, r.batch_size, r.constraint, r.wall_time_s_mean, r.wall_time_s_std, r.wall_time_s_min, r.cold_time_s, r.max_violation, r.cold_violation], ","))
        end
    end
    println()
    println("Wrote $(length(results)) rows to $OUT_PATH")
end

main()
