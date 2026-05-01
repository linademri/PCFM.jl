#=
Run a named benchmark suite from TOML.

Example:
  julia --project=. benchmarks/run_suite.jl benchmarks/suites/small_cpu.toml
=#

using PCFM
include(joinpath(@__DIR__, "BenchmarkUtils.jl"))
using .BenchmarkUtils

if length(ARGS) != 1
    error("Usage: julia --project=. benchmarks/run_suite.jl benchmarks/suites/<suite>.toml")
end

suite_path = ARGS[1]
suite = read_suite(suite_path)
name = get(suite, "name", splitext(basename(suite_path))[1])
outdir = get(suite, "outdir", joinpath("results", name))
mkpath(outdir)

function csv_join(x)
    if x isa AbstractVector
        return join(x, ",")
    end
    return string(x)
end

base_env = Dict{String, Any}()
for key in ["nx", "nt", "batches", "constraint", "backends", "warmup", "repeat", "seed", "steps", "tol"]
    if haskey(suite, key)
        envkey = "PCFM_BENCH_" * uppercase(key)
        base_env[envkey] = csv_join(suite[key])
    end
end

benchmarks = get(suite, "benchmarks", ["projection_deep", "sample_quality"])

for bench in benchmarks
    env = copy(base_env)
    if bench == "projection_deep"
        env["PCFM_BENCH_OUT"] = joinpath(outdir, "projection_deep.csv")
        run_script_with_env(joinpath(@__DIR__, "benchmark_projection_deep.jl"), env)
    elseif bench == "sample_quality"
        env["PCFM_BENCH_OUT"] = joinpath(outdir, "sample_quality.csv")
        if haskey(suite, "reference_backend")
            env["PCFM_BENCH_REFERENCE_BACKEND"] = suite["reference_backend"]
        end
        run_script_with_env(joinpath(@__DIR__, "benchmark_sample_quality.jl"), env)
    elseif bench == "projection"
        env["PCFM_BENCH_OUT"] = joinpath(outdir, "projection.csv")
        run_script_with_env(joinpath(@__DIR__, "benchmark_projection.jl"), env)
    elseif bench == "sample_pcfm"
        env["PCFM_BENCH_OUT"] = joinpath(outdir, "sample_pcfm.csv")
        run_script_with_env(joinpath(@__DIR__, "benchmark_sample_pcfm.jl"), env)
    elseif bench == "madnlp_sparsity"
        env["PCFM_BENCH_OUT"] = joinpath(outdir, "madnlp_sparsity.csv")
        run_script_with_env(joinpath(@__DIR__, "benchmark_madnlp_sparsity.jl"), env)
    else
        error("Unknown benchmark in suite: $bench")
    end
end
