"""
    AbstractProjectionSolver

Defines how the constrained least-squares projection

```
min_Z ‖Z − Ẑ‖²_F  s.t.  h(Zᵢ) = 0,  i = 1, …, Nᵦ
```

is solved. The batch structure is block-diagonal (each sample's constraint is independent),
so any reasonable solver exploits that.

Subtypes must implement:

```julia
project!(solver, Z, Ẑ, constraint) -> Z
```

where `Z, Ẑ` are `(nx, nt, 1, Nᵦ)` arrays. `Z` is written in place; `Ẑ` is the unconstrained
pre-projection point. By convention the solver should work out-of-place safely too (i.e.
passing `Z === Ẑ` is allowed; internally the solver copies as needed).

The guarantee on return: `‖h(Zᵢ)‖ ≤ solver.tol` for each sample, or the solver's maximum
iteration count was hit (in which case it should warn). Correctness of constraint
satisfaction is the solver's responsibility; choice of where on the constraint manifold to
land is shared with the formulation (least-squares to `Ẑ`).

The interface intentionally does not take batched Jacobians as a parameter — each solver
chooses how to exploit block-diagonal structure (per-sample loop, vmap-style batching, or
a fused symbolic formulation). Swapping solvers at the PCFM call site should be transparent.
"""
abstract type AbstractProjectionSolver end

"""
    project!(solver, Z, Ẑ, constraint) -> Z

Solve the batched projection and write the result into `Z`. Default implementation falls
back to per-sample projection via `project_sample!`, which is what most solvers want anyway
given the block-diagonal structure. Solvers that fuse the whole batch (e.g. MadNLP/ExaModels)
override `project!` directly.
"""
function project!(
    solver::AbstractProjectionSolver,
    Z::AbstractArray{T, 4},
    Ẑ::AbstractArray{T, 4},
    constraint::AbstractConstraint,
) where {T}
    nx, nt, _, Nb = size(Ẑ)
    n = nx * nt

    # Flatten to (n, Nb) for per-sample iteration. We use reshape (no copy) so mutations
    # through the flattened view propagate back to Z.
    Ẑ_flat = reshape(Ẑ, n, Nb)
    Z_flat = reshape(Z, n, Nb)

    for i in 1:Nb
        ẑ_i = @view Ẑ_flat[:, i]
        z_i = @view Z_flat[:, i]
        project_sample!(solver, z_i, ẑ_i, constraint)
    end
    return Z
end

# Non-mutating convenience form: allocates output, calls the mutating one.
function project(solver::AbstractProjectionSolver, Ẑ::AbstractArray, constraint::AbstractConstraint)
    Z = similar(Ẑ)
    copyto!(Z, Ẑ)
    return project!(solver, Z, Ẑ, constraint)
end

# ---------------------------------------------------------------------------
# BatchedGaussNewtonSolver — reference CPU/GPU-array implementation.
# ---------------------------------------------------------------------------

"""
    BatchedGaussNewtonSolver(; tol=1e-8, max_iter=25, damping=0.0, verbose=false)

Reference Gauss-Newton projection solver. Operates per sample (relying on block-diagonal
structure) and uses ForwardDiff (via the constraint's `jacobian` method) to build `Jᵢ` at
each iterate.

# Algorithm

At each Newton iterate `zₖ`, linearise `h(zₖ + Δ) ≈ h(zₖ) + Jₖ Δ` and solve

```
min_Δ  ½ ‖(zₖ + Δ) − ẑ‖²    s.t.   Jₖ Δ = −h(zₖ)
```

The KKT conditions give, for `r := zₖ − ẑ` and `g := h(zₖ) + Jₖ r`,

```
Δ = −r  − Jₖᵀ (Jₖ Jₖᵀ + λI)⁻¹ g,
zₖ₊₁ = zₖ + Δ = ẑ − Jₖᵀ (…)⁻¹ g + …
```

For linear constraints this converges in a single step; for nonlinear constraints (energy,
mass with nonlinear transport, etc.) convergence is quadratic in the neighbourhood of the
manifold, which is exactly where PCFM's correction step lives.

# Fields

- `tol`:       constraint residual tolerance `‖h(z)‖∞`
- `max_iter`:  maximum Newton iterations before bailing with a warning
- `damping`:   Levenberg-Marquardt style `λ` added to `J Jᵀ` for conditioning; 0 is standard GN
- `verbose`:   print per-iteration residuals for debugging

# Notes on scale

With the per-sample `m × m` linear solve (where `m` is the constraint dimension, typically 1
or `nx` for IC), each Newton iteration is O(m² n + m³). For `Nb = 256, nx = nt = 100, m = 1`,
each sweep is ~10⁴ flops/sample × 256 samples = ~2.5 Mflops per Newton step. Negligible next
to the FNO model forward pass. The bottleneck is Jacobian construction (ForwardDiff over
`n`-dim input), which is why override methods on `jacobian` matter when `m ≪ n`.
"""
struct BatchedGaussNewtonSolver <: AbstractProjectionSolver
    tol::Float64
    max_iter::Int
    damping::Float64
    verbose::Bool
end

function BatchedGaussNewtonSolver(; tol = 1e-8, max_iter = 25, damping = 0.0, verbose = false)
    return BatchedGaussNewtonSolver(tol, max_iter, damping, verbose)
end

"""
    project_sample!(solver, z, ẑ, constraint) -> z

Per-sample Newton loop. `z` is both input (warm start) and output; caller should initialise
`z .= ẑ` (the default `project!` does this via `copyto!`). Returns `z`.
"""
function project_sample!(
    solver::BatchedGaussNewtonSolver,
    z::AbstractVector,
    ẑ::AbstractVector,
    constraint::AbstractConstraint,
)
    # Fast path: linear constraint ⇒ a single GN step is exact (up to roundoff).
    # We still run the loop but break after one iteration in practice.
    max_iter = is_linear(constraint) ? 1 : solver.max_iter
    λ = solver.damping

    for k in 1:max_iter
        h = residual(constraint, z)
        rnorm = maximum(abs, h)
        if solver.verbose
            @info "GN iter" k residual = rnorm
        end
        if rnorm ≤ solver.tol
            break
        end

        J = jacobian(constraint, z)
        m = size(J, 1)

        # KKT solve: Δ = −r − Jᵀ (J Jᵀ + λI)⁻¹ (h − J r), with r = z − ẑ.
        r = z .- ẑ
        g = h .- J * r
        A = J * transpose(J)
        if λ > 0
            @inbounds for i in 1:m
                A[i, i] += λ
            end
        end
        # Small m × m solve. `\` dispatches to LAPACK (LU/Cholesky) for dense; for a 1×1 it
        # is a scalar divide. This is the hot loop for scalar constraints and is cheap.
        y = A \ g
        Δ = .-(r) .- transpose(J) * y

        z .+= Δ

        if k == solver.max_iter && !is_linear(constraint)
            @warn "BatchedGaussNewton did not converge" final_residual=rnorm tol=solver.tol
        end
    end
    return z
end
