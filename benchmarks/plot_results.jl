#=
plot_results.jl

Turn benchmark CSVs into comparison plots. Kept deliberately simple — Plots.jl + StatsPlots.jl,
one figure per benchmark type.

Usage:
    julia --project=benchmarks benchmarks/plot_results.jl results/*.csv

Produces PNG files next to each input CSV.
=#

using CSV
using DataFrames
using Plots
using StatsPlots
using Printf

gr()  # headless-friendly backend

function plot_projection(df::DataFrame, out_path::String)
    # Assumes columns: backend, batch_size, wall_time_s_mean, wall_time_s_std, max_violation.
    plt = plot(
        title = "Projection wall time vs. batch size",
        xlabel = "Batch size (Nb)",
        ylabel = "Wall time (s, warm)",
        xscale = :log10,
        yscale = :log10,
        legend = :topleft,
        dpi = 150,
    )
    for g in groupby(df, :backend)
        backend = first(g.backend)
        plot!(plt,
            g.batch_size, g.wall_time_s_mean;
            yerror = g.wall_time_s_std,
            label = string(backend),
            marker = :circle,
            linewidth = 2,
        )
    end
    savefig(plt, out_path)
    @printf "Wrote %s\n" out_path

    # Second plot: constraint violation.
    plt2 = plot(
        title = "Max constraint violation (should be at solver tol)",
        xlabel = "Batch size (Nb)",
        ylabel = "max |h(z)|",
        xscale = :log10,
        yscale = :log10,
        legend = :topleft,
        dpi = 150,
    )
    for g in groupby(df, :backend)
        backend = first(g.backend)
        plot!(plt2,
            g.batch_size, g.max_violation;
            label = string(backend),
            marker = :square,
            linewidth = 2,
        )
    end
    viol_path = replace(out_path, ".png" => "_violation.png")
    savefig(plt2, viol_path)
    @printf "Wrote %s\n" viol_path
end

function plot_madnlp_sparsity(df::DataFrame, out_path::String)
    # Side-by-side bars: structured vs unstructured solve time.
    batches = sort(unique(df.batch_size))
    structured   = [df[(df.batch_size .== b) .& (df.variant .== "structured"),   :solve_s][1] for b in batches]
    unstructured = [df[(df.batch_size .== b) .& (df.variant .== "unstructured"), :solve_s][1] for b in batches]

    plt = groupedbar(
        [structured unstructured];
        labels = ["structured (SIMD-exploiting)" "unstructured"],
        xlabel = "Batch size",
        ylabel = "Solve time (s)",
        xticks = (1:length(batches), string.(batches)),
        title = "MadNLP: block-diagonal vs. flat formulation",
        yscale = :log10,
        dpi = 150,
    )
    savefig(plt, out_path)
    @printf "Wrote %s\n" out_path
end

function detect_and_plot(csv_path::String)
    df = CSV.read(csv_path, DataFrame)
    out_path = replace(csv_path, r"\.csv$" => ".png")

    # Heuristic dispatch based on columns present.
    cols = Set(names(df))
    if "variant" in cols && "iters" in cols
        plot_madnlp_sparsity(df, out_path)
    elseif "backend" in cols && "wall_time_s_mean" in cols
        plot_projection(df, out_path)
    else
        @warn "Don't know how to plot $csv_path (columns: $(names(df)))"
    end
end

if isempty(ARGS)
    println("Usage: julia plot_results.jl <csv_path>...")
    exit(1)
end

for path in ARGS
    if isfile(path)
        @printf "Plotting %s\n" path
        detect_and_plot(path)
    else
        @warn "Skipping non-file: $path"
    end
end
