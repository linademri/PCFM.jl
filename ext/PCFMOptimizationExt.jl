"""
    PCFMOptimizationExt

Weak-dependency extension loaded when both PCFM and Optimization are available. Provides
`OptimizationJLSolver`, which wraps Optimization.jl's unified interface so users can pick
LBFGS, IPNewton, or any other Optimization.jl-compatible method as the projection backend.

This is a solver-agnostic baseline: correctness over performance. For production GPU work
with built-in constraints, use `MadNLPGPUSolver`; for a minimal Jacobian-accurate CPU
reference or any arbitrary user residual, use `BatchedGaussNewtonSolver`.

# Formulation

Rather than solve the constrained QP directly, we minimise a quadratic penalty

```
(1/2) ‖z − ẑ‖² + (ρ/2) ‖h(z)‖²
```

with `ρ` increasing across outer iterations (a simple quadratic penalty method). This maps
cleanly onto Optimization.jl's `OptimizationProblem` without needing explicit Lagrange
multiplier machinery, at the cost of being less accurate for tight tolerances than a true
constrained solver.

For users who need exact constraint satisfaction, prefer the Gauss-Newton or Reactant
solvers. This extension is primarily an interface-validation baseline.
"""
module PCFMOptimizationExt

using PCFM
using Optimization
using OptimizationOptimJL: LBFGS

import PCFM: AbstractProjectionSolver, project_sample!, residual

# The concrete type `_OptimizationJLSolverImpl` is declared in the core package as a stub
# subtype of AbstractProjectionSolver, so `using PCFM` alone shows the name. Here in the
# extension we provide the user-facing constructor `OptimizationJLSolver(;kwargs...)` which
# returns an instance of the stub, and the `project_sample!` implementation that actually
# uses Optimization.jl.

function PCFM.OptimizationJLSolver(;
    tol = 1e-6,
    max_iter = 200,
    ρ_init = 1e2,
    ρ_mult = 10.0,
    ρ_max = 1e8,
    outer_iter = 5,
    optimizer = LBFGS(),
)
    return PCFM._OptimizationJLSolverImpl(tol, max_iter, ρ_init, ρ_mult, ρ_max, outer_iter, optimizer)
end

function project_sample!(
    solver::PCFM._OptimizationJLSolverImpl,
    z::AbstractVector,
    ẑ::AbstractVector,
    constraint::PCFM.AbstractConstraint,
)
    ρ = solver.ρ_init
    # Objective with parameter p = (ρ,). `u` is the decision vector.
    function loss(u, p)
        ρ_ = p[1]
        h = residual(constraint, u)
        return 0.5 * sum(abs2, u .- ẑ) + 0.5 * ρ_ * sum(abs2, h)
    end

    optf = OptimizationFunction(loss, Optimization.AutoForwardDiff())
    u0 = copy(z)

    for _ in 1:solver.outer_iter
        prob = OptimizationProblem(optf, u0, [ρ])
        sol = solve(prob, solver.optimizer; maxiters = solver.max_iter)
        u0 .= sol.u
        h = residual(constraint, u0)
        if maximum(abs, h) ≤ solver.tol
            break
        end
        ρ = min(ρ * solver.ρ_mult, solver.ρ_max)
    end

    z .= u0
    return z
end

# End of extension — the constructor and project_sample! methods above are automatically
# loaded once both PCFM and Optimization are present in the session.

end # module
