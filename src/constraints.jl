"""
    AbstractConstraint

A constraint of the form `h(z) = 0` where `z` is a single sample (not a batch).
Concrete subtypes must implement:

- `residual(c, z)::AbstractVector`  — returns `h(z)` of length `m` (the number of scalar constraints)
- `constraint_dim(c)::Int`          — returns `m`

Optionally, they may specialise:

- `jacobian(c, z)::AbstractMatrix`  — returns `∂h/∂z` of size `(m, n)`; falls back to AD.
- `is_linear(c)::Bool`               — defaults to `false`; when `true` enables fast paths.

`z` is treated flatly: the solver reshapes the `(nx, nt, 1)` tensor for a single sample
into a length-`nx*nt` vector before calling `residual`/`jacobian`. Constraint authors can
reshape internally to recover the spatial/temporal layout — see `EnergyConservationConstraint`.

The batched dimension is handled by the solver, not the constraint — a constraint describes
a single sample's physics, and the solver maps it over the batch.
"""
abstract type AbstractConstraint end

"""
    residual(c::AbstractConstraint, z::AbstractVector) -> AbstractVector

Return `h(z)` as a length-`m` vector. Must be differentiable.
"""
function residual end

"""
    constraint_dim(c::AbstractConstraint) -> Int

Return `m`, the number of scalar equality constraints. For an energy or mass constraint this
is 1. For a boundary-condition constraint fixing `k` points, this is `k`.
"""
function constraint_dim end

"""
    is_linear(c::AbstractConstraint) -> Bool

Return `true` if `h(z)` is affine in `z`. Defaults to `false`. Linear constraints admit a
single-step closed-form projection, so solvers can skip Newton iteration.
"""
is_linear(::AbstractConstraint) = false

"""
    jacobian(c::AbstractConstraint, z::AbstractVector) -> AbstractMatrix

Return `∂h/∂z` at `z`, size `(m, n)`. The default falls back to `ForwardDiff.jacobian`, which
is adequate for `m, n` in the hundreds. Override for large `n` or when a closed form exists.
"""
function jacobian(c::AbstractConstraint, z::AbstractVector)
    # Import locally so the core package does not hard-depend on ForwardDiff.
    # In practice ForwardDiff is used by BatchedGaussNewtonSolver and is listed as a dep.
    return ForwardDiff.jacobian(zi -> residual(c, zi), z)
end

# ---------------------------------------------------------------------------
# Built-in constraints
# ---------------------------------------------------------------------------

"""
    LinearICConstraint(u0; nx, nt)

Fix the first spatial slice of a `(nx, nt, 1)` sample to a prescribed initial condition
`u0::AbstractVector` of length `nx`. Reproduces the original PCFM.jl behaviour of

```julia
@. x_1[:, 1:1, :, :] = u_0_ic
```

as a formal projection. `h(z) = z[1:nx, 1] - u0`, so `m = nx`, `J` is a constant selector
matrix (ones on the IC indices, zeros elsewhere). Linear, so no Newton iteration required.
"""
struct LinearICConstraint{V <: AbstractVector} <: AbstractConstraint
    u0::V
    nx::Int
    nt::Int
end

constraint_dim(c::LinearICConstraint) = c.nx
is_linear(::LinearICConstraint) = true

function residual(c::LinearICConstraint, z::AbstractVector)
    # z is laid out as vec(reshape(z, nx, nt)), so z[1:nx] corresponds to the first column,
    # which is the t = 0 slice (first nt index? depends on storage).
    #
    # The original code indexes `x_1[:, 1:1, :, :]` — first `nt` index, all `nx`. With
    # column-major storage and reshape order (nx, nt), that slice is stored at indices
    # 1:nx of the flattened vector. We respect that convention throughout.
    return @view(z[1:c.nx]) .- c.u0
end

function jacobian(c::LinearICConstraint, z::AbstractVector)
    # Selector: J[i, i] = 1 for i in 1:nx, 0 otherwise.
    n = c.nx * c.nt
    J = zeros(eltype(z), c.nx, n)
    @inbounds for i in 1:c.nx
        J[i, i] = one(eltype(z))
    end
    return J
end

"""
    EnergyConservationConstraint(E0; nx, nt, dx = 1/nx)

Scalar nonlinear constraint `∫ (1/2) u(x, T)² dx = E₀` on the final-time slice of the
sample. `T` is the last `nt` index (i.e. `z[(nt-1)*nx+1 : nt*nx]` in flattened form).
This is a canonical benchmark for PCFM because energy is quadratic — nonlinear but cheap —
and the Jacobian has a clean closed form.

`h(z) = (dx / 2) * Σᵢ u(xᵢ, T)² - E₀`, so `m = 1` and `J_j = dx * u(x_j, T)` on the final-slice
indices and `0` elsewhere.
"""
struct EnergyConservationConstraint{T <: Real} <: AbstractConstraint
    E0::T
    nx::Int
    nt::Int
    dx::T
end

function EnergyConservationConstraint(E0::Real; nx::Int, nt::Int, dx = 1 / nx)
    T = promote_type(typeof(E0), typeof(dx))
    return EnergyConservationConstraint{T}(T(E0), nx, nt, T(dx))
end

constraint_dim(::EnergyConservationConstraint) = 1

function residual(c::EnergyConservationConstraint, z::AbstractVector)
    nx, nt = c.nx, c.nt
    # Final-time slice: columns nt-1 (in 0-indexed) ⇒ indices (nt-1)*nx+1 : nt*nx.
    u_T = @view z[((nt - 1) * nx + 1):(nt * nx)]
    e = (c.dx / 2) * sum(abs2, u_T)
    return [e - c.E0]
end

function jacobian(c::EnergyConservationConstraint, z::AbstractVector)
    nx, nt = c.nx, c.nt
    n = nx * nt
    J = zeros(eltype(z), 1, n)
    final_range = ((nt - 1) * nx + 1):(nt * nx)
    @inbounds for (k, j) in enumerate(final_range)
        J[1, j] = c.dx * z[j]
    end
    return J
end

"""
    MassConservationConstraint(m0; nx, nt, dx = 1/nx)

Scalar linear constraint `∫ u(x, T) dx = m₀`. Included separately from `LinearICConstraint`
because it fixes an integral, not a slice. Linear in `z`, so the projection is a single
closed-form step — but we keep it as a general constraint to exercise the code path with a
dense row Jacobian rather than a selector.
"""
struct MassConservationConstraint{T <: Real} <: AbstractConstraint
    m0::T
    nx::Int
    nt::Int
    dx::T
end

function MassConservationConstraint(m0::Real; nx::Int, nt::Int, dx = 1 / nx)
    T = promote_type(typeof(m0), typeof(dx))
    return MassConservationConstraint{T}(T(m0), nx, nt, T(dx))
end

constraint_dim(::MassConservationConstraint) = 1
is_linear(::MassConservationConstraint) = true

function residual(c::MassConservationConstraint, z::AbstractVector)
    nx, nt = c.nx, c.nt
    u_T = @view z[((nt - 1) * nx + 1):(nt * nx)]
    return [c.dx * sum(u_T) - c.m0]
end

function jacobian(c::MassConservationConstraint, z::AbstractVector)
    nx, nt = c.nx, c.nt
    n = nx * nt
    J = zeros(eltype(z), 1, n)
    J[1, ((nt - 1) * nx + 1):(nt * nx)] .= c.dx
    return J
end

"""
    NonlinearConstraint(h, m; jac = nothing)

Wrap an arbitrary user-supplied residual `h(z)::AbstractVector` of output length `m`. If `jac`
is supplied it is used directly; otherwise `jacobian` falls back to ForwardDiff. This is the
escape hatch for users who want to encode a bespoke PDE boundary condition or nonlinear
invariant without subtyping `AbstractConstraint` themselves.
"""
struct NonlinearConstraint{F, J} <: AbstractConstraint
    h::F
    m::Int
    jac::J  # Either `nothing` or a function `z -> AbstractMatrix`.
end

NonlinearConstraint(h, m::Int; jac = nothing) = NonlinearConstraint(h, m, jac)

constraint_dim(c::NonlinearConstraint) = c.m
residual(c::NonlinearConstraint, z::AbstractVector) = c.h(z)

function jacobian(c::NonlinearConstraint, z::AbstractVector)
    if c.jac === nothing
        return ForwardDiff.jacobian(c.h, z)
    else
        return c.jac(z)
    end
end
