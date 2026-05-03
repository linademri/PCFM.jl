#=
benchmark_madnlp_sparsity.jl

MadNLP-specific benchmark answering the proposal's question:
  "Sparsity exploitation: KKT factorization time and iteration count for MadNLP under
   block-diagonal vs. unstructured formulation."

We solve the same batched projection problem two ways and compare MadNLP statistics:

  1. Structured: ExaCore with variables z[i, j] (2-index), constraint generator indexed
     by (i, j) or (j,) — ExaModels sees the block-diagonal structure and emits one
     derivative kernel shared across all j. This is the intended usage.

  2. Unstructured: variables z[k] flattened to a single index, constraint generator
     over k with the j-structure hidden. Same mathematical problem, but ExaModels now
     sees a single "flat" constraint pattern and cannot factor out the per-sample
     repetition; the resulting Jacobian assembly is less efficient even though MadNLP's
     eventual linear-solver step may be similar.

The comparison isolates the ExaModels SIMD-abstraction benefit specifically. MadNLP's
linear solver sees block-diagonal sparsity in both cases (that's a property of the
underlying problem, not of how it's written), so the delta is in AD cost — which is
exactly what the ExaModels paper claims is the lever.

Usage:
    julia --project=. benchmarks/benchmark_madnlp_sparsity.jl

Env vars: PCFM_BENCH_BATCHES, PCFM_BENCH_NX, PCFM_BENCH_NT, PCFM_BENCH_CONSTRAINT, PCFM_BENCH_OUT.
=#

using PCFM
using ExaModels
using MadNLP
using Random
using Printf

const BATCHES    = parse.(Int, split(get(ENV, "PCFM_BENCH_BATCHES", "32,64,128"), ","))
const NX         = parse(Int, get(ENV, "PCFM_BENCH_NX", "32"))    # smaller default — this is a structural diagnostic
const NT         = parse(Int, get(ENV, "PCFM_BENCH_NT", "32"))
const CONSTRAINT = Symbol(get(ENV, "PCFM_BENCH_CONSTRAINT", "energy"))
const OUT_PATH   = get(ENV, "PCFM_BENCH_OUT", "results_madnlp_sparsity.csv")
const SEED       = parse(Int, get(ENV, "PCFM_BENCH_SEED", "20260420"))

Random.seed!(SEED)

# ----- Structured (block-diagonal-exposing) formulation -----------------------

function build_structured(Ẑ::Matrix{Float64}, kind::Symbol, nx::Int, nt::Int)
    n, Nb = size(Ẑ)
    c = ExaCore()
    z = variable(c, n, Nb; start = (Ẑ[i, j] for i = 1:n, j = 1:Nb))

    obj_refs = [(i, j, Ẑ[i, j]) for i = 1:n, j = 1:Nb]
    objective(c, 0.5 * (z[r[1], r[2]] - r[3])^2 for r in obj_refs)

    if kind === :energy
        E0 = 1.0
        dx = 1 / nx
        fs = (nt - 1) * nx + 1
        fe = nt * nx
        constraint(c, sum(0.5 * dx * z[i, j]^2 for i = fs:fe) - E0 for j = 1:Nb)
    elseif kind === :mass
        m0 = 0.0
        dx = 1 / nx
        fs = (nt - 1) * nx + 1
        fe = nt * nx
        constraint(c, sum(dx * z[i, j] for i = fs:fe) - m0 for j = 1:Nb)
    else
        error("Unsupported constraint kind: $kind")
    end
    return ExaModel(c)
end

# ----- Unstructured formulation -----------------------------------------------
#
# Same problem, but we flatten z to a single 1-D index of length n*Nb and unroll the
# per-sample constraint manually. ExaModels sees N_b distinct constraint patterns (one
# per sample) rather than one pattern repeated N_b times — this is the inefficient
# authoring pattern we want to compare against.

function build_unstructured(Ẑ::Matrix{Float64}, kind::Symbol, nx::Int, nt::Int)
    n, Nb = size(Ẑ)
    N = n * Nb
    c = ExaCore()
    z = variable(c, N; start = (Ẑ[mod1(k, n), div(k - 1, n) + 1] for k = 1:N))
    # Objective is the same quadratic, over flattened indexing.
    obj_refs = [(k, Ẑ[mod1(k, n), div(k - 1, n) + 1]) for k = 1:N]

    objective(c, 0.5 * (z[r[1]] - r[2])^2 for r in obj_refs)
    # Per-sample constraint as a separate `constraint(...)` call per j. This defeats
    # ExaModels' SIMD abstraction because each call sees only one constraint expression
    # rather than a generator over j.
    dx = 1 / nx
    fs_local = (nt - 1) * nx + 1
    fe_local = nt * nx
    if kind === :energy
        E0 = 1.0
        for j in 1:Nb
            offset = (j - 1) * n
            constraint(c, sum(0.5 * dx * z[offset + i]^2 for i = fs_local:fe_local) - E0)
        end
    elseif kind === :mass
        m0 = 0.0
        for j in 1:Nb
            offset = (j - 1) * n
            constraint(c, sum(dx * z[offset + i] for i = fs_local:fe_local) - m0)
        end
    else
        error("Unsupported constraint kind: $kind")
    end
    return ExaModel(c)
end

# ----- Solve + report helpers ------------------------------------------------

function solve_and_report(model, label::String)
    # `madnlp(...)` returns a solver result with detailed statistics; we extract the ones
    # the proposal explicitly asks about.
    result = madnlp(model; tol = 1e-8, print_level = MadNLP.ERROR)
    # Different MadNLP versions expose stats differently; we collect defensively.
    stats = (
        label = label,
        iterations = hasproperty(result, :iter) ? result.iter :
                     hasproperty(result, :iterations) ? result.iterations : -1,
        total_time = hasproperty(result, :counters) ? result.counters.total_time : -1.0,
        status = hasproperty(result, :status) ? string(result.status) : "UNKNOWN",
        objective = hasproperty(result, :objective) ? result.objective : NaN,
    )
    return stats, result
end

function bench_one(Nb::Int, kind::Symbol)
    Ẑ = Matrix{Float64}(randn(NX * NT, Nb))

    @printf "  Nb = %d: " Nb

    # Structured
    print("structured...")
    t_build_s = @elapsed m_s = build_structured(Ẑ, kind, NX, NT)
    stats_s, _ = solve_and_report(m_s, "structured")
    @printf " build = %.3f s, solve = %.3f s, iters = %d;  " t_build_s stats_s.total_time stats_s.iterations

    # Unstructured
    print("unstructured...")
    t_build_u = @elapsed m_u = build_unstructured(Ẑ, kind, NX, NT)
    stats_u, _ = solve_and_report(m_u, "unstructured")
    @printf " build = %.3f s, solve = %.3f s, iters = %d\n" t_build_u stats_u.total_time stats_u.iterations

    return [
        (Nb = Nb, variant = "structured",   build_s = t_build_s, solve_s = stats_s.total_time, iters = stats_s.iterations, status = stats_s.status),
        (Nb = Nb, variant = "unstructured", build_s = t_build_u, solve_s = stats_u.total_time, iters = stats_u.iterations, status = stats_u.status),
    ]
end

println("=" ^ 70)
println("MadNLP sparsity-structure benchmark")
println("=" ^ 70)
@printf "  nx = %d, nt = %d, constraint = %s\n" NX NT CONSTRAINT
println()

all_rows = []
for Nb in BATCHES
    append!(all_rows, bench_one(Nb, CONSTRAINT))
end

open(OUT_PATH, "w") do io
    println(io, "batch_size,variant,build_s,solve_s,iters,status")
    for r in all_rows
        println(io, join([r.Nb, r.variant, r.build_s, r.solve_s, r.iters, r.status], ","))
    end
end
println("\nWrote $(length(all_rows)) rows to $OUT_PATH")
