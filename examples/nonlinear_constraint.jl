"""
Example: nonlinear constraint projection with PCFM.jl.

This script demonstrates the new `AbstractConstraint` / `AbstractProjectionSolver` API
introduced in PCFM 0.2. We focus on the *projection subproblem* in isolation, because that
is the novel extension — the sampling loop around it is unchanged and covered in
`train_diffusion.jl`.

Three things are shown:

 1. Enforcing an energy-conservation constraint, `∫ ½ u(x,T)² dx = E₀`, on a batch of
    arbitrary noise fields via `BatchedGaussNewtonSolver` (pure Julia, CPU or GPU).
 2. If `ExaModels` and `MadNLP` are available, the same projection via `MadNLPSolver`,
    with a cross-check that both backends agree to float tolerance.
 3. Verification that constraint residuals are at/near float tolerance.

Run:

```
julia --project examples/nonlinear_constraint.jl
```
"""

using PCFM
using LinearAlgebra
using Random
using Printf
using Statistics: mean

Random.seed!(42)

# ----- Configuration ---------------------------------------------------------

const nx = 100
const nt = 100
const Nb = 32         # batch size
const E0 = 1.0f0      # target energy

println("="^60)
println("Nonlinear Projection Benchmark (Energy Conservation)")
println("="^60)
println("nx = $nx   nt = $nt   batch = $Nb   E₀ = $E0")
println()

# ----- 1. Build the constraint and a synthetic unprojected batch -------------

constraint = EnergyConservationConstraint(E0; nx = nx, nt = nt)

# Random candidate samples — the kind of `x_1_hat` that would come out of the extrapolation
# step in `sample_pcfm`, except we do not need a trained FNO to exercise the projection.
Ẑ = randn(Float32, nx, nt, 1, Nb)

# Pre-projection energies (per sample).
function energies(Z, constraint)
    nx, nt, _, Nb = size(Z)
    dx = 1 / nx
    return [(dx / 2) * sum(abs2, Z[:, nt, 1, i]) for i in 1:Nb]
end

E_pre = energies(Ẑ, constraint)
@printf "Pre-projection energy range: [%.3g, %.3g]\n" minimum(E_pre) maximum(E_pre)
@printf "Mean absolute deviation from E₀: %.3g\n\n" mean(abs.(E_pre .- E0))

# ----- 2. Gauss-Newton baseline ---------------------------------------------

println("[Gauss-Newton] BatchedGaussNewtonSolver")
solver_gn = BatchedGaussNewtonSolver(tol = 1e-7, max_iter = 25)
t0 = time()
Z_gn = project(solver_gn, Ẑ, constraint)
t_gn = time() - t0

E_post_gn = energies(Z_gn, constraint)
@printf "  Wall time:            %.3f s\n" t_gn
@printf "  Max |h(z)| post-proj: %.3g\n" maximum(abs.(E_post_gn .- E0))
@printf "  Mean ‖z − ẑ‖₂:        %.3g\n\n" mean(norm(vec(Z_gn[:, :, 1, i] .- Ẑ[:, :, 1, i])) for i in 1:Nb)

# ----- 3. MadNLP cross-check (optional) --------------------------------------
#
# Only runs if ExaModels and MadNLP are loadable. On the same projection problem, MadNLP
# should land on the same point on the constraint manifold (energy is a convex quadratic
# equality — minimum-norm projection is unique on each of the two branches of the manifold;
# both solvers start from Ẑ so they pick the same branch).

const HAS_MADNLP = try
    @eval using ExaModels, MadNLP
    true
catch
    false
end

if HAS_MADNLP
    println("[MadNLP] MadNLPSolver (cross-check)")
    solver_mn = MadNLPSolver(tol = 1e-8)

    print("  Building + solving... ")
    t0 = time()
    Z_mn = project(solver_mn, Ẑ, constraint)
    t_mn = time() - t0
    @printf "done in %.3f s\n" t_mn

    E_post_mn = energies(Z_mn, constraint)
    @printf "  Max |h(z)| post-proj: %.3g\n" maximum(abs.(E_post_mn .- E0))
    @printf "  Mean ‖z − ẑ‖₂:        %.3g\n\n" mean(norm(vec(Z_mn[:, :, 1, i] .- Ẑ[:, :, 1, i])) for i in 1:Nb)

    # Backend agreement.
    max_disagreement = maximum(abs.(Z_gn .- Z_mn))
    @printf "Max pointwise disagreement between GN and MadNLP: %.3g\n" max_disagreement
    @printf "(Small values — ~1e-5 in Float32 — indicate both backends land on the\n"
    @printf " same point on the energy manifold, as expected.)\n"
else
    println("[MadNLP cross-check skipped — install ExaModels + MadNLP to enable]")
    println("  julia> using Pkg; Pkg.add([\"ExaModels\", \"MadNLP\"])")
end
