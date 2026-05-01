#=
Generate a lightweight Markdown evaluation report from benchmark CSVs.

Run with the benchmarks environment because this script needs CSV/DataFrames:
  julia --project=benchmarks benchmarks/make_report.jl results/**/*.csv
=#

using CSV
using DataFrames
using Dates
using Printf
using Statistics

if isempty(ARGS)
    error("Usage: julia --project=benchmarks benchmarks/make_report.jl results/*.csv")
end

out_path = get(ENV, "PCFM_BENCH_REPORT_OUT", "docs/evaluation.md")
mkpath(dirname(out_path) == "" ? "." : dirname(out_path))

function read_any(path)
    try
        return DataFrame(CSV.File(path)), path
    catch e
        @warn "Skipping unreadable CSV" path exception=e
        return DataFrame(), path
    end
end

pairs = [read_any(p) for p in ARGS]
frames = [p[1] for p in pairs if nrow(p[1]) > 0]
paths = [p[2] for p in pairs if nrow(p[1]) > 0]
isempty(frames) && error("No readable CSV inputs")

data = reduce(vcat, frames; cols = :union)

function md_table(df::DataFrame; maxrows::Int = 20)
    if nrow(df) == 0
        return "_No rows._\n"
    end
    showdf = first(df, min(maxrows, nrow(df)))
    cols = names(showdf)
    io = IOBuffer()
    println(io, "| ", join(cols, " | "), " |")
    println(io, "| ", join(fill("---", length(cols)), " | "), " |")
    for row in eachrow(showdf)
        vals = [replace(string(row[c]), "|" => "\\|") for c in cols]
        println(io, "| ", join(vals, " | "), " |")
    end
    if nrow(df) > maxrows
        println(io, "\n_Showing first $maxrows of $(nrow(df)) rows._")
    end
    return String(take!(io))
end

function maybe_round!(df::DataFrame)
    for c in names(df)
        if eltype(df[!, c]) <: AbstractFloat
            df[!, c] = round.(df[!, c]; sigdigits = 5)
        end
    end
    return df
end

open(out_path, "w") do io
    println(io, "# PCFM.jl Evaluation Report")
    println(io)
    println(io, "Generated: ", now())
    println(io)
    println(io, "## Inputs")
    for p in paths
        println(io, "- `", p, "`")
    end
    println(io)

    if "benchmark" in names(data)
        println(io, "## Benchmarks covered")
        println(io, md_table(combine(groupby(data, :benchmark), nrow => :rows)))
    end

    if all(c -> c in names(data), ["backend", "batch_size", "wall_time_s_mean"])
        println(io, "## Projection scaling")
        df = combine(groupby(dropmissing(data, [:backend, :batch_size, :wall_time_s_mean]), [:backend, :batch_size]),
            :wall_time_s_mean => mean => :wall_time_s_mean)
        sort!(df, [:backend, :batch_size])
        println(io, md_table(maybe_round!(df)))
    end

    if all(c -> c in names(data), ["backend", "batch_size", "median_violation", "p95_violation", "max_violation", "failure_rate"])
        println(io, "## Constraint violation distribution")
        df = select(dropmissing(data, [:backend, :batch_size, :median_violation, :p95_violation, :max_violation, :failure_rate]),
            :backend, :batch_size, :median_violation, :p95_violation, :max_violation, :failure_rate)
        sort!(df, [:backend, :batch_size])
        println(io, md_table(maybe_round!(df)))
    end

    if all(c -> c in names(data), ["backend", "batch_size", "mmd_rbf", "energy_distance", "mean_l2", "cov_frobenius", "spectral_l2"])
        println(io, "## Sample/distribution quality")
        df = select(dropmissing(data, [:backend, :batch_size, :mmd_rbf]),
            :backend, :batch_size, :mmd_rbf, :energy_distance, :mean_l2, :cov_frobenius, :spectral_l2)
        sort!(df, [:backend, :batch_size])
        println(io, md_table(maybe_round!(df)))
    end

    if all(c -> c in names(data), ["backend", "batch_size", "status"])
        println(io, "## Backend statuses")
        df = combine(groupby(data, [:backend, :status]), nrow => :rows)
        sort!(df, [:backend, :status])
        println(io, md_table(df))
    end

    println(io, "## Interpretation checklist")
    println(io, "- Check `failure_rate` before comparing speed.")
    println(io, "- Compare warm timing separately from cold timing/manifest setup cost.")
    println(io, "- Use distribution metrics as diagnostics; for final paper results, run the same schema on model samples versus held-out PDE trajectories.")
end

@info "wrote report" out_path
