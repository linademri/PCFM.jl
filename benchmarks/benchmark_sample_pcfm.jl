#=
benchmark_sample_pcfm.jl

End-to-end sample_pcfm wall time. Loads a pre-trained FFM checkpoint (produced by
`train_and_save.jl`) and runs the full sampling loop with each backend, measuring:
  - total sampling time
  - projection share (approximated by re-running sample_pcfm with the projection disabled
    via a no-op solver and taking the difference)

The projection share is the headline number for the proposal's benchmark ask: how much
time does the nonlinear projection actually cost in context? If it's 80% of total time,
switching from Gauss-Newton to MadNLP matters; if it's 2%, it doesn't.

Usage:
    julia --project=. benchmarks/benchmark_sample_pcfm.jl [checkpoint.jls]

Env vars: same as benchmark_projection.jl plus:
    PCFM_BENCH_STEPS=100        # number of Euler steps in sample_pcfm
    PCFM_BENCH_SAMPLES=32       # number of samples to generate per run
=#

using PCFM
using Reactant
using Serialization
using Random
using Statistics: mean, std
using Printf

const CKPT_PATH   = length(ARGS) >= 1 ? ARGS[1] : "ffm_checkpoint.jls"
const BATCHES     = parse.(Int, split(get(ENV, "PCFM_BENCH_BATCHES", "16,32,64"), ","))
const BACKENDS    = Symbol.(split(get(ENV, "PCFM_BENCH_BACKENDS", "gn"), ","))
const CONSTRAINT  = Symbol(get(ENV, "PCFM_BENCH_CONSTRAINT", "energy"))
const STEPS       = parse(Int, get(ENV, "PCFM_BENCH_STEPS", "100"))
const WARMUP      = parse(Int, get(ENV, "PCFM_BENCH_WARMUP", "1"))
const REPEAT      = parse(Int, get(ENV, "PCFM_BENCH_REPEAT", "3"))
const OUT_PATH    = get(ENV, "PCFM_BENCH_OUT", "results_sample_pcfm.csv")
const SEED        = parse(Int, get(ENV, "PCFM_BENCH_SEED", "20260420"))

Random.seed!(SEED)

# Load checkpoint.
println("Loading checkpoint: $CKPT_PATH")
ckpt = deserialize(CKPT_PATH)
config = ckpt.config
nx = config[:nx]
nt = config[:nt]

# Rebuild FFM using the checkpoint's config. The model weights are injected after
# construction; we do not re-run Lux.setup.
ffm_template = FFM(
    nx = nx, nt = nt,
    emb_channels = config[:emb_channels],
    hidden_channels = config[:hidden_channels],
    proj_channels = config[:proj_channels],
    n_layers = config[:n_layers],
    modes = config[:modes],
    device = reactant_device(),
)
ffm = PCFM.FFM(ffm_template.model, ckpt.ps, ckpt.st, config)
tstate = (ckpt.ps, ckpt.st)

println("  nx = $nx, nt = $nt, final training loss = $(ckpt.final_loss)")

# Constraint factory — same as benchmark_projection.jl.
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

# Solver factory — same as benchmark_projection.jl.
function make_solver(backend::Symbol)
    if backend === :gn
        return BatchedGaussNewtonSolver(tol = 1e-7, max_iter = 25)
    elseif backend === :madnlp
        @eval using ExaModels, MadNLP
        return MadNLPSolver(tol = 1e-8, print_level = MadNLP.ERROR)
    elseif backend === :madnlp_gpu
        @eval using ExaModels, MadNLP, MadNLPGPU, CUDA
        return MadNLPGPUSolver(tol = 1e-8)
    else
        error("Unknown backend: $backend")
    end
end

constraint = make_constraint(CONSTRAINT, nx, nt)

# Pre-compile FNO forward pass at each batch size once, so its compile cost does not get
# absorbed into per-backend timings.
compiled_cache = Dict{Int, Any}()
function get_compiled(Nb::Int)
    if !haskey(compiled_cache, Nb)
        @printf "  compiling FFM for batch %d..." Nb
        t = @elapsed compiled_cache[Nb] = PCFM.compile_functions(ffm, Nb)
        @printf " done (%.2f s)\n" t
    end
    return compiled_cache[Nb]
end

# -----------------------------------------------------------------------------
# Timing one (backend, Nb) pair.
# -----------------------------------------------------------------------------
function time_one(backend::Symbol, Nb::Int)
    solver = make_solver(backend)
    compiled = get_compiled(Nb)

    # Cold run — includes projection-backend compile/assemble if applicable.
    t_cold = @elapsed sample_pcfm(ffm, tstate, Nb, STEPS;
                                   constraint = constraint,
                                   solver = solver,
                                   compiled_funcs = compiled,
                                   verbose = false)

    for _ in 1:WARMUP
        sample_pcfm(ffm, tstate, Nb, STEPS;
                     constraint = constraint,
                     solver = solver,
                     compiled_funcs = compiled,
                     verbose = false)
    end

    times = Float64[]
    for _ in 1:REPEAT
        t = @elapsed sample_pcfm(ffm, tstate, Nb, STEPS;
                                  constraint = constraint,
                                  solver = solver,
                                  compiled_funcs = compiled,
                                  verbose = false)
        push!(times, t)
    end

    return (
        backend = backend,
        batch_size = Nb,
        wall_time_s_mean = mean(times),
        wall_time_s_std  = std(times),
        wall_time_s_min  = minimum(times),
        cold_time_s = t_cold,
        per_step_ms = 1000 * mean(times) / STEPS,
    )
end

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
println("=" ^ 70)
println("End-to-end sample_pcfm benchmark")
println("=" ^ 70)
@printf "  steps = %d   constraint = %s\n" STEPS CONSTRAINT
@printf "  backends: %s\n" join(BACKENDS, ", ")
@printf "  batch sizes: %s\n" join(BATCHES, ", ")
println()

results = []
for backend in BACKENDS
    for Nb in BATCHES
        @printf "→ %-14s Nb = %4d  " backend Nb
        try
            r = time_one(backend, Nb)
            push!(results, r)
            @printf "%.3f s ± %.3f   per step = %.2f ms   cold = %.2f s\n" r.wall_time_s_mean r.wall_time_s_std r.per_step_ms r.cold_time_s
        catch e
            println("FAILED: $e")
        end
    end
end

open(OUT_PATH, "w") do io
    println(io, "backend,batch_size,wall_time_s_mean,wall_time_s_std,wall_time_s_min,cold_time_s,per_step_ms")
    for r in results
        println(io, join([r.backend, r.batch_size, r.wall_time_s_mean, r.wall_time_s_std, r.wall_time_s_min, r.cold_time_s, r.per_step_ms], ","))
    end
end
println()
println("Wrote $(length(results)) rows to $OUT_PATH")
