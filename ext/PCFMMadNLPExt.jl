"""
    PCFMMadNLPExt

Weak-dependency extension loaded when both `ExaModels` and `MadNLP` are present. Implements
`MadNLPSolver`, a projection backend that assembles the entire batched projection QP as a
single `ExaModel` and hands it to MadNLP.

# Why this is worth it

The projection subproblem at each PCFM correction step is

```
min_{Z ∈ ℝ^{n × Nb}}  ½ ‖Z − Ẑ‖²_F    s.t.   h(Z[:, b]) = 0   for b = 1..Nb
```

This is `Nb` independent equality-constrained QPs stacked together. Naively, one would loop
over samples and solve each small NLP separately — but that misses the batch structure. The
ExaModels SIMD abstraction is designed for exactly this pattern: you write the constraint
as a generator expression indexed by sample, ExaModels infers the repeated structure, and
derivative evaluation (gradient, Jacobian, Hessian) is done with one custom kernel per
repeated pattern rather than a walk over a generic expression tree.

MadNLP then sees a single NLP with a block-diagonal Jacobian, and its condensed-space
solver produces all `Nb` solutions simultaneously. On GPU (via `PCFMMadNLPGPUExt`) the
factorisation goes through cuDSS. On CPU we get MadNLP's default linear solver (Umfpack
or similar, depending on what is available).

# Interface

```julia
using PCFM, ExaModels, MadNLP
solver = MadNLPSolver(tol = 1e-8, print_level = MadNLP.ERROR)
Z = project(solver, Ẑ, constraint)
```

# Constraint support

Each `AbstractConstraint` subtype needs an `_add_constraints!(core, z, constraint, Nb)`
method that injects its equality constraints into the ExaModels core. Built-ins: linear IC,
energy, mass. Generic `NonlinearConstraint` is not currently supported through this path
because ExaModels' expression language is restricted — the residual must be written in
terms of ExaModels-traceable ops (see `_add_constraints!` methods below for the pattern).
Users who have a PDE-specific constraint should add a method in their own code.
"""
module PCFMMadNLPExt

using PCFM
using ExaModels
using MadNLP

import PCFM: AbstractProjectionSolver, project!,
             LinearICConstraint, EnergyConservationConstraint, MassConservationConstraint,
             AbstractConstraint

# -------- Constructor (activated once both deps are loaded) -----------------

function PCFM.MadNLPSolver(;
    tol = 1e-8,
    max_iter = 200,
    print_level = MadNLP.ERROR,
    backend = nothing,   # pass CUDABackend() via PCFMMadNLPGPUExt for GPU
    linear_solver = nothing,
)
    return PCFM._MadNLPSolverImpl(tol, max_iter, print_level, backend, linear_solver)
end

# -------- Build the ExaModels core for a given constraint ------------------

"""
    _build_exa_core(Ẑ::Matrix, constraint, backend) -> (ExaCore, variable_handle)

Construct the ExaCore with:
  - variable `z[i, j]` of shape `(n, Nb)`
  - objective `½ Σ (z[i,j] - Ẑ[i,j])²`
  - constraints from the dispatching `_add_constraints!`

# A note on ExaModels data access

ExaModels generators do not let the body close over external Julia arrays like `Ẑ[i, j]` —
when the generator is traced, `i` and `j` are ExaModels index *tokens*, not `Int`s, so
`getindex(Ẑ, i, j)` on a plain `Matrix` raises an `ArgumentError`. The documented idiom is
to pack external data into an iterator of tuples (which ExaModels *does* understand how to
destructure inside the trace) and index those tuple fields with literal integers:

```julia
refs = [(i, j, Ẑ[i, j]) for i = 1:n, j = 1:Nb]
objective(c, 0.5 * (z[r[1], r[2]] - r[3])^2 for r in refs)
```

Here `r` is a traced tuple token, and `r[1]`/`r[2]`/`r[3]` are recognised by ExaModels as
constant field accesses, so they lower to the SIMD kernel correctly. Same idiom applies to
`LinearICConstraint` (which needs `u0[i]`).
"""
function _build_exa_core(Ẑ::AbstractMatrix, constraint::AbstractConstraint, backend)
    n, Nb = size(Ẑ)

    c = backend === nothing ? ExaCore() : ExaCore(; backend = backend)

    # Variable with warm-start = Ẑ. No bounds — the projection problem has no box constraints.
    z = variable(c, n, Nb; start = (Ẑ[i, j] for i = 1:n, j = 1:Nb))

    # Objective: ½ ‖Z − Ẑ‖²_F. We cannot close over `Ẑ[i, j]` directly inside the ExaModels
    # generator body (see docstring). Pack (i, j, Ẑ_ij) into tuples and destructure.
    obj_refs = [(i, j, Ẑ[i, j]) for i = 1:n, j = 1:Nb]
    objective(c, 0.5 * (z[r[1], r[2]] - r[3])^2 for r in obj_refs)

    # Constraints — dispatch on constraint type.
    _add_constraints!(c, z, constraint, Nb)

    return c, z
end

# -------- Constraint-specific equality generators --------------------------

function _add_constraints!(c, z, con::LinearICConstraint, Nb::Int)
    nx = con.nx
    u0 = con.u0
    # Equality: z[i, j] - u0[i] = 0 for i in 1:nx, j in 1:Nb.
    # Same external-data trick as in the objective: pack (i, j, u0[i]) into tuples rather
    # than close over the plain Julia `u0` vector inside the trace.
    ic_refs = [(i, j, u0[i]) for i = 1:nx, j = 1:Nb]
    constraint(c, z[r[1], r[2]] - r[3] for r in ic_refs)
    return c
end

function _add_constraints!(c, z, con::EnergyConservationConstraint, Nb::Int)
    nx, nt = con.nx, con.nt
    dx = con.dx
    E0 = con.E0
    # Scalar constraint per sample: (dx/2) Σ_i z[i_final, j]^2 − E0 = 0.
    # ExaModels lets us express the per-sample sum via `sum(... for i = ...)` inside the
    # generator body. The outer generator over `j` produces Nb scalar constraints, all of
    # which share the same algebraic form — ExaModels vectorises them.
    final_start = (nt - 1) * nx + 1
    final_stop = nt * nx
    constraint(
        c,
        sum(0.5 * dx * z[i, j]^2 for i = final_start:final_stop) - E0
        for j = 1:Nb
    )
    return c
end

function _add_constraints!(c, z, con::MassConservationConstraint, Nb::Int)
    nx, nt = con.nx, con.nt
    dx = con.dx
    m0 = con.m0
    final_start = (nt - 1) * nx + 1
    final_stop = nt * nx
    constraint(
        c,
        sum(dx * z[i, j] for i = final_start:final_stop) - m0
        for j = 1:Nb
    )
    return c
end

function _add_constraints!(::Any, ::Any, con::AbstractConstraint, ::Int)
    error("PCFMMadNLPExt: no _add_constraints! method for $(typeof(con)). See the docstring " *
          "of PCFMMadNLPExt for how to add one — the body must be a pure arithmetic " *
          "expression expressible in ExaModels' restricted modeling language.")
end

# -------- project! override ------------------------------------------------

function project!(
    solver::PCFM._MadNLPSolverImpl,
    Z::AbstractArray{T, 4},
    Ẑ::AbstractArray{T, 4},
    constraint::AbstractConstraint,
) where {T}
    nx, nt, _, Nb = size(Ẑ)
    n = nx * nt

    # ExaModels' default numeric type is Float64. Going all-Float32 requires a
    # ExaCore(T = Float32) (supported but less well-tested). For now we promote on entry and
    # cast back on exit, which matches the FFM Float32 convention at the API boundary.
    Ẑ_flat = Matrix{Float64}(reshape(Ẑ, n, Nb))

    core, _ = _build_exa_core(Ẑ_flat, constraint, solver.backend)
    model = ExaModel(core)

    madnlp_kwargs = Dict{Symbol, Any}(
        :tol => solver.tol,
        :max_iter => solver.max_iter,
        :print_level => solver.print_level,
    )
    if solver.linear_solver !== nothing
        madnlp_kwargs[:linear_solver] = solver.linear_solver
    end

    result = madnlp(model; madnlp_kwargs...)
    # `result.solution` is a flat length-(n*Nb) vector in column-major order matching the
    # `(i, j)` iteration of the variable generator.
    sol = reshape(result.solution, n, Nb)

    copyto!(reshape(Z, n, Nb), T.(sol))
    return Z
end

end # module