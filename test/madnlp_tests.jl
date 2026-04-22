# Tests for the PCFMMadNLPExt extension. Gated on ExaModels + MadNLP being loadable in the
# current session; when unavailable, the testset is skipped with a notice rather than
# failing CI.
#
# The tests compare MadNLP's projected solution against BatchedGaussNewton's solution on
# the same inputs. For convex equality-constrained QPs (which all three built-in
# constraints produce at the linearised level, and Energy is truly convex quadratic in z
# via its level set) the minimiser is unique, so the two backends should agree to the
# combined tolerances of each solver.

using Test
using PCFM
using PCFM: project, LinearICConstraint, EnergyConservationConstraint,
            MassConservationConstraint, BatchedGaussNewtonSolver
using Random
using LinearAlgebra

const HAS_MADNLP = try
    @eval using ExaModels
    @eval using MadNLP
    true
catch
    false
end

if !HAS_MADNLP
    @info "Skipping PCFMMadNLPExt tests — ExaModels/MadNLP not available"
else
    # Note on naming: both PCFM and MadNLP export a symbol named `MadNLPSolver`. PCFM's is
    # the projection-backend constructor we want to test; MadNLP's is an unrelated core
    # IPM solver type. Because we `using MadNLP` at top to check availability, the bare
    # name `MadNLPSolver` is ambiguous inside the tests, so we qualify every call site as
    # `PCFM.MadNLPSolver(...)`.

    # The PCFMMadNLPExt extension is loaded eagerly by Julia's package extensions mechanism
    # as soon as `using ExaModels, MadNLP` succeeds. At that point `PCFM.MadNLPSolver(...)`
    # returns a real `_MadNLPSolverImpl` rather than erroring.

    const NX = 12
    const NT = 8
    const NB = 4

    Random.seed!(7777)

    @testset "MadNLP backend: LinearICConstraint" begin
        u0 = Float64.(randn(NX))
        c = LinearICConstraint(u0, NX, NT)
        Ẑ = randn(NX, NT, 1, NB)

        # Reference projection via Gauss-Newton.
        solver_ref = BatchedGaussNewtonSolver(tol = 1e-10)
        Z_ref = project(solver_ref, Ẑ, c)

        # MadNLP projection. `print_level = ERROR` suppresses solver chatter; tol kept tight.
        # Note: `MadNLPSolver` is qualified with `PCFM.` because MadNLP.jl also exports a
        # type with that same name, and the two collide when both `using PCFM` and
        # `using MadNLP` are active in the test environment.
        solver_mn = PCFM.MadNLPSolver(tol = 1e-9)
        Z_mn = project(solver_mn, Ẑ, c)

        # First slice must match u0 to solver tol.
        for i in 1:NB
            @test maximum(abs, Z_mn[:, 1, 1, i] .- u0) < 1e-6
        end
        # Solution should agree with the Gauss-Newton reference.
        @test maximum(abs, Z_mn .- Z_ref) < 1e-5
    end

    @testset "MadNLP backend: MassConservationConstraint" begin
        m0 = 0.3
        c = MassConservationConstraint(m0; nx = NX, nt = NT)
        Ẑ = randn(NX, NT, 1, NB)

        solver_ref = BatchedGaussNewtonSolver(tol = 1e-10)
        Z_ref = project(solver_ref, Ẑ, c)

        solver_mn = PCFM.MadNLPSolver(tol = 1e-9)
        Z_mn = project(solver_mn, Ẑ, c)

        dx = 1 / NX
        for i in 1:NB
            @test abs(dx * sum(Z_mn[:, NT, 1, i]) - m0) < 1e-6
        end
        @test maximum(abs, Z_mn .- Z_ref) < 1e-5
    end

    @testset "MadNLP backend: EnergyConservationConstraint" begin
        E0 = 0.5
        c = EnergyConservationConstraint(E0; nx = NX, nt = NT)
        Ẑ = randn(NX, NT, 1, NB)

        solver_ref = BatchedGaussNewtonSolver(tol = 1e-10, max_iter = 50)
        Z_ref = project(solver_ref, Ẑ, c)

        solver_mn = PCFM.MadNLPSolver(tol = 1e-9)
        Z_mn = project(solver_mn, Ẑ, c)

        dx = 1 / NX
        for i in 1:NB
            u_T = Z_mn[:, NT, 1, i]
            @test abs((dx / 2) * sum(abs2, u_T) - E0) < 1e-6
        end
        # Energy is nonconvex as an equality constraint (two solutions symmetric about
        # origin), so for this test we only require constraint satisfaction — not that
        # MadNLP and GN land at the same point. The reference projection starts from Ẑ
        # so both solvers should pick the same branch, but we do not want the test to
        # depend on that. So: compare Frobenius distances instead.
        dist_mn = norm(Z_mn .- Ẑ)
        dist_ref = norm(Z_ref .- Ẑ)
        @test abs(dist_mn - dist_ref) / dist_ref < 1e-3
    end
end