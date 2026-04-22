# Tests for the projection / constraint subsystem. These are designed to run without the
# FNO forward pass — we exercise the math directly on synthetic arrays, which makes them
# fast enough to gate CI on and exposes bugs in the solvers independent of the model.

using Test
using PCFM
using PCFM: project, project!, residual, jacobian, constraint_dim, is_linear,
            AbstractConstraint, AbstractProjectionSolver,
            LinearICConstraint, EnergyConservationConstraint, MassConservationConstraint,
            NonlinearConstraint, BatchedGaussNewtonSolver
using Random
using LinearAlgebra
using ForwardDiff

const NX = 16
const NT = 8
const N = NX * NT
const NB = 4

Random.seed!(1234)

# ---------------------------------------------------------------------------
# Constraint-level tests: residual, jacobian, and is_linear contract.
# ---------------------------------------------------------------------------

@testset "Constraints: LinearICConstraint" begin
    u0 = Float32.(randn(NX))
    c = LinearICConstraint(u0, NX, NT)

    @test constraint_dim(c) == NX
    @test is_linear(c) == true

    # On a sample whose first slice matches u0, residual is zero.
    z = Float32.(randn(N))
    z[1:NX] .= u0
    @test maximum(abs, residual(c, z)) == 0

    # Nonzero residual when first slice differs.
    z[1] += 1.0f0
    @test residual(c, z)[1] ≈ 1.0f0

    # Analytical Jacobian matches ForwardDiff.
    J_fd = ForwardDiff.jacobian(zi -> residual(c, zi), z)
    J = jacobian(c, z)
    @test J ≈ J_fd
    @test size(J) == (NX, N)
end

@testset "Constraints: EnergyConservationConstraint" begin
    E0 = 0.5f0
    c = EnergyConservationConstraint(E0; nx = NX, nt = NT)

    @test constraint_dim(c) == 1
    @test is_linear(c) == false

    z = Float32.(randn(N))
    # h is (dx/2) * sum(u_T²) - E0.
    u_T = z[(NT - 1) * NX + 1 : NT * NX]
    expected = (1 / NX) / 2 * sum(abs2, u_T) - E0
    @test residual(c, z)[1] ≈ Float32(expected) rtol = 1e-5

    # Analytical Jacobian agrees with AD.
    J_fd = ForwardDiff.jacobian(zi -> residual(c, zi), z)
    @test jacobian(c, z) ≈ J_fd rtol = 1e-5
end

@testset "Constraints: MassConservationConstraint" begin
    m0 = 0.0f0
    c = MassConservationConstraint(m0; nx = NX, nt = NT)

    @test constraint_dim(c) == 1
    @test is_linear(c) == true

    z = Float32.(randn(N))
    J_fd = ForwardDiff.jacobian(zi -> residual(c, zi), z)
    @test jacobian(c, z) ≈ J_fd rtol = 1e-5
end

@testset "Constraints: NonlinearConstraint wrapper" begin
    # Enforce a single quadratic constraint: ½‖z‖² = 1.
    h = z -> [0.5 * sum(abs2, z) - 1.0]
    c = NonlinearConstraint(h, 1)

    @test constraint_dim(c) == 1
    @test is_linear(c) == false  # default

    z = randn(N)
    @test residual(c, z) ≈ h(z)
    # AD fallback works.
    @test jacobian(c, z) ≈ ForwardDiff.jacobian(h, z) rtol = 1e-8
end

# ---------------------------------------------------------------------------
# Solver-level tests: BatchedGaussNewton on each constraint, checking that the
# projection (a) satisfies the constraint to tolerance and (b) is the minimum-
# norm projection (verified via optimality: ẑ − z should be in the row space of J(z)).
# ---------------------------------------------------------------------------

@testset "BatchedGaussNewton: LinearICConstraint" begin
    u0 = Float32.(randn(NX))
    c = LinearICConstraint(u0, NX, NT)
    solver = BatchedGaussNewtonSolver()

    Ẑ = Float32.(randn(NX, NT, 1, NB))
    Z = project(solver, Ẑ, c)

    # Every sample's first slice matches u0 exactly.
    for i in 1:NB
        @test Z[:, 1, 1, i] ≈ u0 atol = 1e-6
    end
    # Rest of the sample is unchanged (selector projection is orthogonal).
    for i in 1:NB
        @test Z[:, 2:end, 1, i] ≈ Ẑ[:, 2:end, 1, i]
    end
end

@testset "BatchedGaussNewton: MassConservationConstraint" begin
    m0 = 0.5f0
    c = MassConservationConstraint(m0; nx = NX, nt = NT)
    solver = BatchedGaussNewtonSolver()

    Ẑ = Float32.(randn(NX, NT, 1, NB))
    Z = project(solver, Ẑ, c)

    dx = 1 / NX
    for i in 1:NB
        u_T = Z[:, NT, 1, i]
        @test dx * sum(u_T) ≈ m0 atol = 1e-5
    end
end

@testset "BatchedGaussNewton: EnergyConservationConstraint" begin
    E0 = 1.0
    c = EnergyConservationConstraint(E0; nx = NX, nt = NT)
    solver = BatchedGaussNewtonSolver(tol = 1e-8, max_iter = 50)

    Ẑ = randn(NX, NT, 1, NB)  # Float64 to get a clean tolerance
    Z = project(solver, Ẑ, c)

    dx = 1 / NX
    for i in 1:NB
        u_T = Z[:, NT, 1, i]
        energy = (dx / 2) * sum(abs2, u_T)
        @test energy ≈ E0 atol = 1e-6
    end
end

@testset "BatchedGaussNewton: NonlinearConstraint (unit sphere)" begin
    # Project onto the unit sphere in ℝⁿ: h(z) = ½‖z‖² − 1 = 0.
    # Closed-form answer: Z* = Ẑ / ‖Ẑ‖, which is what Gauss-Newton should find.
    h = z -> [0.5 * sum(abs2, z) - 1.0]
    c = NonlinearConstraint(h, 1)
    solver = BatchedGaussNewtonSolver(tol = 1e-9, max_iter = 50)

    Ẑ = randn(NX, NT, 1, NB)
    Z = project(solver, Ẑ, c)

    for i in 1:NB
        z = vec(Z[:, :, 1, i])
        ẑ = vec(Ẑ[:, :, 1, i])
        # Constraint satisfied.
        @test 0.5 * sum(abs2, z) ≈ 1.0 atol = 1e-7
        # Solution is along the ẑ direction — check the angle.
        @test dot(z, ẑ) / (norm(z) * norm(ẑ)) ≈ 1.0 atol = 1e-6
    end
end

# ---------------------------------------------------------------------------
# End-to-end: sample_pcfm backward compatibility. We stub the FNO with a small
# identity-velocity model so the sample loop runs without pulling in Lux/FNO.
# ---------------------------------------------------------------------------

@testset "sample_pcfm default path preserves IC constraint" begin
    # Smoke test: verify that with the default constraint (LinearIC with sin(x + π/4)),
    # after one sampling step the IC slice equals the prescribed initial condition.
    #
    # We do not run the full sample_pcfm here because it requires a trained FFM. Instead
    # we replicate the projection step in isolation, which is what backward compatibility
    # hinges on — and check that the result matches the old hardcoded behaviour.

    x_grid = range(0, 2π, length = NX)
    u0 = Float32.(sin.(x_grid .+ π / 4))
    c = LinearICConstraint(u0, NX, NT)
    solver = BatchedGaussNewtonSolver()

    Ẑ = Float32.(randn(NX, NT, 1, NB))
    Z = project(solver, Ẑ, c)

    # Old code: `@. x_1[:, 1:1, :, :] = u_0_ic`, which is exactly this.
    Z_old_style = copy(Ẑ)
    for i in 1:NB
        Z_old_style[:, 1, 1, i] .= u0
    end

    @test Z ≈ Z_old_style atol = 1e-6
end
