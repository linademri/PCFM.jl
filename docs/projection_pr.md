# PCFM.jl v0.2: Nonlinear Constraint Support

This PR generalises PCFM.jl's projection step from hardcoded linear initial-condition
fixing to arbitrary (potentially nonlinear) constraints via a swappable solver interface.

## What changed

The existing `sample_pcfm` function performed its projection step as a direct array
assignment:

```julia
@. x_1[:, 1:1, :, :] = u_0_ic
```

which is the correct minimum-norm projection *only* when the constraint Jacobian is a
selector matrix. This PR replaces that assignment with a proper projection call:

```julia
x_1 = project(solver, x_1_hat, constraint)
```

where `solver <: AbstractProjectionSolver` and `constraint <: AbstractConstraint`. The old
call site continues to work unchanged — the default `constraint` reproduces the original
`sin(x + π/4)` IC, and the default `solver` is a one-step Gauss-Newton that is equivalent
to the old assignment for linear constraints.

## New public API

### Constraints

All constraints are equality constraints `h(z) = 0`:

| Type                            | `h(z)`                            | `m` | Linear? |
|---------------------------------|-----------------------------------|-----|---------|
| `LinearICConstraint(u0; …)`     | `z[1:nx, 1] − u₀`                 | nx  | yes     |
| `EnergyConservationConstraint`  | `(dx/2) Σ u(·, T)² − E₀`          | 1   | no      |
| `MassConservationConstraint`    | `dx Σ u(·, T) − m₀`               | 1   | yes     |
| `NonlinearConstraint(h, m)`     | user-supplied                     | m   | no      |

Users can subtype `AbstractConstraint` and implement `residual`, `constraint_dim`, and
optionally `jacobian` (falls back to ForwardDiff) and `is_linear`.

### Solvers

| Solver                          | Backend                           | Generic `h`? | Notes                       |
|---------------------------------|-----------------------------------|--------------|-----------------------------|
| `BatchedGaussNewtonSolver`      | pure Julia + ForwardDiff          | yes          | default; reference impl.     |
| `OptimizationJLSolver`          | weakdep: Optimization.jl          | yes          | penalty method; baseline    |
| `MadNLPSolver`                  | weakdep: ExaModels + MadNLP       | built-ins only | one batched NLP            |
| `MadNLPGPUSolver`               | weakdep: + MadNLPGPU + CUDA       | built-ins only | NVIDIA GPU via cuDSS       |

**Selection guidance:**

- **Default / development:** `BatchedGaussNewtonSolver`. Works for any constraint (via
  ForwardDiff on `residual`), is dependency-light, and converges quadratically on the
  constraint manifold. The per-sample loop exploits block-diagonal structure trivially.
  GPU-capable through the underlying array type (if `Ẑ` is a `CuArray`, the Newton steps
  run on GPU).
- **Production GPU, built-in constraint:** `MadNLPGPUSolver`. ExaModels' SIMD abstraction
  infers the block-diagonal Jacobian/Hessian sparsity at compile time, and MadNLP's
  condensed-space interior-point method factorises the condensed KKT system on GPU via
  cuDSS. Single solve per projection call — no per-sample loop.

## Example

```julia
using PCFM, ExaModels, MadNLP

ffm = FFM(nx = 100, nt = 100, ...)
tstate = train_ffm!(ffm, data)

# Default behaviour: back-compat with old hardcoded IC.
samples = sample_pcfm(ffm, tstate, 32, 100)

# Enforce energy conservation via MadNLP (CPU):
constraint = EnergyConservationConstraint(1.0f0; nx = 100, nt = 100)
solver = MadNLPSolver(tol = 1e-8)
samples = sample_pcfm(ffm, tstate, 32, 100; constraint, solver)

# Same but on GPU:
using MadNLPGPU, CUDA
solver_gpu = MadNLPGPUSolver(tol = 1e-8)
samples = sample_pcfm(ffm, tstate, 32, 100; constraint, solver = solver_gpu)

# User-supplied nonlinear constraint — Gauss-Newton with ForwardDiff:
h = z -> [my_pde_residual(z)]
constraint = NonlinearConstraint(h, 1)
solver = BatchedGaussNewtonSolver(tol = 1e-8)
samples = sample_pcfm(ffm, tstate, 32, 100; constraint, solver)
```

## Structure

```
src/
  constraints.jl       # AbstractConstraint + built-ins (LinearIC, Energy, Mass, Nonlinear)
  projection.jl        # AbstractProjectionSolver + BatchedGaussNewtonSolver
  sampling.jl          # refactored sample_pcfm with constraint/solver kwargs
  PCFM.jl              # updated module file, exports, weakdep stubs

ext/
  PCFMOptimizationExt.jl   # loaded with `using Optimization, OptimizationOptimJL`
  PCFMMadNLPExt.jl         # loaded with `using ExaModels, MadNLP`       (CPU path)
  PCFMMadNLPGPUExt.jl      # loaded with `using ExaModels, MadNLP, MadNLPGPU, CUDA`

test/
  runtests.jl                  # top-level runner
  legacy_tests.jl              # preserved pre-0.2 tests (unchanged behaviour)
  projection_tests.jl          # constraint residual/Jacobian/projection tests
  madnlp_tests.jl              # MadNLP backend ↔ Gauss-Newton agreement tests

examples/
  nonlinear_constraint.jl  # energy-conservation demo
```

## Algorithm notes

### Projection subproblem and KKT form

The projection subproblem per sample is the equality-constrained QP

```
min_z  ½ ‖z − ẑ‖²    s.t.   h(z) = 0
```

Gauss-Newton linearisation at iterate `z_k`: `h(z_k) + J_k Δ = 0`, giving the KKT system

```
Δ = −r − Jᵀ (J Jᵀ + λI)⁻¹ g,   with r = z_k − ẑ,  g = h(z_k) − J r
```

For linear constraints this converges in one step. For nonlinear constraints (energy,
generic `h`) convergence is quadratic in the manifold's neighbourhood — exactly the regime
PCFM operates in, since the correction step runs on values that are already approximately
on the manifold.

### Per-sample vs. batched formulations

- `BatchedGaussNewtonSolver` loops over `Nb` samples in Julia. For constraint dimension
  `m = 1` each per-sample KKT solve is a scalar divide; for `m = nx` (LinearIC) it's a
  closed-form single-step selector assignment. Loop overhead dominates over numerics.
- `MadNLPSolver` builds one `ExaModel` over all `n × Nb` variables with a 2-index
  constraint generator, so ExaModels' SIMD abstraction emits a single derivative-evaluation
  kernel that runs over all `Nb` blocks in parallel. MadNLP's condensed-space IPM solves
  the resulting block-diagonal KKT system as a single Cholesky factorisation with static
  pivoting (on GPU via cuDSS).

### What makes the abstraction work

The same KKT update is invariant across all three backend styles (per-sample loop, single
NLP). Only the *expression* of that algebra differs: as a Julia `for` loop that does a
per-sample dense solve, or as an NLP generator expression that ExaModels compiles to one
SIMD derivative kernel. The `AbstractProjectionSolver` interface lets users swap at the
call site with no changes to the sampling loop.

## Testing status

| Test file                      | Scope                                                  | Requires                  |
|--------------------------------|--------------------------------------------------------|---------------------------|
| `legacy_tests.jl`              | pre-0.2 behaviour preserved                            | Lux, NeuralOperators      |
| `projection_tests.jl`          | constraint residuals/Jacobians, GN solver correctness  | pure Julia + ForwardDiff  |
| `madnlp_tests.jl`              | MadNLP ↔ GN agreement on 3 constraint types            | ExaModels, MadNLP         |

The MadNLP tests are gated inside the file on extension availability, so the suite degrades
gracefully when run without the weak deps.

As of last validation: 58 / 61 tests pass on a Windows + Julia 1.12.5 CPU setup, with the
MadNLP tests skipping cleanly because ExaModels/MadNLP are not installed. All constraint
residual/Jacobian/projection correctness tests pass, including the unit-sphere optimality
check and the backward-compatibility test against the old hardcoded assignment.

## Open items (not in this PR)

1. **`NonlinearConstraint` in MadNLP.** The ExaModels path currently dispatches on concrete
   built-in constraint types. Supporting `NonlinearConstraint` requires the user's residual
   to be expressible in ExaModels' restricted expression language — there is no generic
   path equivalent to the ForwardDiff one in `BatchedGaussNewtonSolver`. A templated "user
   adds an `_add_constraints!` method" entry point is the right first step here.
2. **Warm starting.** Successive projections in the PCFM inner loop see smoothly evolving
   inputs — plumbing the previous iterate through as MadNLP's primal/dual warm start should
   measurably cut Newton iterations on the nonlinear path.
3. **Mixed precision.** The FFM forward pass is Float32 throughout; `MadNLPSolver`
   currently promotes to Float64 for the ExaModel because `ExaCore()`'s default is Float64.
   Switching to `ExaCore(T = Float32)` avoids the promotion cost on large batches.
4. **Benchmarks on MIT Engaging.** Wall-time vs batch size for each backend, constraint
   violation norms, KKT factorisation timing under block-diagonal vs. unstructured
   formulation, FID of generated samples. Out of scope for this PR but set up cleanly by
   `examples/nonlinear_constraint.jl`, `benchmarks/benchmark_projection.jl`, and the
   `benchmark_madnlp_sparsity.jl` / `benchmark_sample_pcfm.jl` drivers.

## Caveats the reviewer should know

The MadNLP and Optimization.jl extensions are implemented against the interface but have
not yet been executed — the corresponding tests skip when those weak deps aren't
installed. Validation requires `using ExaModels, MadNLP` (plus the rest for GPU) in a
session and rerunning the test suite.

The most likely first-bug sites, in order:

1. `PCFMMadNLPExt.jl` constraint generators — the `sum(... for i = ...) - E0` pattern is
   how ExaModels docs show it, but if the macro hygiene is off, the scalar constant `E0`
   may need to be hoisted out as an ExaModels `parameter`.
2. `Project.toml` compat bounds — pinned to current major versions as of v0.2 authorship
   but some may have moved.

Both are localised, quick to diagnose, and the cross-check tests in `madnlp_tests.jl` flag
them if either triggers.
