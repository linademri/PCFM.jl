"""
    sample_ffm(ffm::FFM, tstate, n_samples, n_steps; kwargs...)

Generate samples from the trained Functional Flow Matching model using Euler integration.
Unconstrained — i.e. no projection step. See `sample_pcfm` for the physics-constrained
variant.

# Arguments

  - `ffm`: FFM model
  - `tstate`: Training state (or a `(ps, st)` tuple)
  - `n_samples`: Number of samples to generate
  - `n_steps`: Number of Euler integration steps
  - `use_compiled`: Whether to use compiled functions
  - `compiled_funcs`: Compiled functions from `compile_functions`
  - `verbose`: Print progress

# Returns

  - Generated samples of shape `(nx, nt, 1, n_samples)`
"""
function sample_ffm(ffm::FFM, tstate, n_samples, n_steps;
        use_compiled = true,
        compiled_funcs = nothing,
        verbose = true)
    nx = ffm.config[:nx]
    nt = ffm.config[:nt]
    emb_channels = ffm.config[:emb_channels]
    device = ffm.config[:device]

    ps, st = _unpack_tstate(tstate)

    if use_compiled && compiled_funcs !== nothing
        model_fn = compiled_funcs.model
        prepare_input_fn = compiled_funcs.prepare_input
    else
        model_fn = ffm.model
        prepare_input_fn = prepare_input
    end

    x = randn(Float32, nx, nt, 1, n_samples) |> device
    dt = 1.0f0 / n_steps

    for step in 0:(n_steps - 1)
        if verbose && step % 10 == 0
            println("Sampling step: $step/$n_steps")
        end

        t_scalar = step * dt
        t_vec = fill(t_scalar, n_samples) |> device
        x_input = prepare_input_fn(x, t_vec, nx, nt, n_samples, emb_channels)
        v, st = model_fn(x_input, ps, st)
        x = x .+ v .* dt
    end

    return x
end

# Helper: extract (ps, st) from either a TrainState or a (ps, st) tuple. Factored out to
# avoid duplication between sample_ffm and sample_pcfm.
function _unpack_tstate(tstate)
    if hasfield(typeof(tstate), :parameters)
        return tstate.parameters, tstate.states
    else
        return tstate[1], tstate[2]
    end
end

"""
    sample_pcfm(ffm::FFM, tstate, n_samples, n_steps;
                constraint = nothing, solver = nothing,
                use_compiled = true, compiled_funcs = nothing, verbose = true)

Physics-Constrained Flow Matching sampling. At each Euler step the velocity field is used
to extrapolate to `t = 1`, the extrapolated state is projected onto the constraint manifold
`{z : h(z) = 0}`, and the corrected endpoint is re-interpolated back to the current time.

# Arguments

  - `ffm`, `tstate`, `n_samples`, `n_steps`: as in `sample_ffm`.
  - `constraint::AbstractConstraint`: physical constraint to enforce. If `nothing`, defaults
    to the original hardcoded IC `u(x, 0) = sin(x + π/4)` to preserve behaviour of the
    pre-0.2 API.
  - `solver::AbstractProjectionSolver`: projection backend. If `nothing`, defaults to
    `BatchedGaussNewtonSolver()` — the reference implementation. For large batches with a
    built-in constraint, `MadNLPSolver()` (CPU) or `MadNLPGPUSolver()` (NVIDIA GPU) solves
    the whole batched QP as one block-diagonal NLP via ExaModels + MadNLP.
  - `use_compiled`, `compiled_funcs`, `verbose`: as in `sample_ffm`.

# Returns

  - Generated samples of shape `(nx, nt, 1, n_samples)`, each satisfying
    `‖h(zᵢ)‖ ≤ solver.tol`.

# Examples

Preserve the old behaviour exactly:

```julia
samples = sample_pcfm(ffm, tstate, 32, 100; compiled_funcs)
```

Enforce an energy-conservation constraint on the final-time slice:

```julia
using ExaModels, MadNLP
constraint = EnergyConservationConstraint(1.0f0; nx, nt)
solver = MadNLPSolver(tol = 1e-8)
samples = sample_pcfm(ffm, tstate, 32, 100; constraint, solver, compiled_funcs)
```
"""
function sample_pcfm(ffm::FFM, tstate, n_samples, n_steps;
        constraint::Union{Nothing, AbstractConstraint} = nothing,
        solver::Union{Nothing, AbstractProjectionSolver} = nothing,
        use_compiled = true,
        compiled_funcs = nothing,
        verbose = true)
    nx = ffm.config[:nx]
    nt = ffm.config[:nt]
    emb_channels = ffm.config[:emb_channels]
    device = ffm.config[:device]

    ps, st = _unpack_tstate(tstate)

    if use_compiled && compiled_funcs !== nothing
        model_fn = compiled_funcs.model
        prepare_input_fn = compiled_funcs.prepare_input
    else
        model_fn = ffm.model
        prepare_input_fn = prepare_input
    end

    # Back-compat default: the original `sin(x + π/4)` IC constraint.
    if constraint === nothing
        x_grid = range(0, 2π, length = nx)
        u0 = Float32.(sin.(x_grid .+ π / 4))
        constraint = LinearICConstraint(u0, nx, nt)
    end

    # Default solver: the Gauss-Newton reference. It handles linear constraints in a single
    # step via the `is_linear` fast path, so the default path remains as cheap as the old
    # hardcoded assignment (modulo a constant factor).
    if solver === nothing
        solver = BatchedGaussNewtonSolver()
    end

    x_0 = randn(Float32, nx, nt, 1, n_samples) |> device
    x = copy(x_0)
    dt = 1.0f0 / n_steps

    for step in 0:(n_steps - 1)
        if verbose && step % 10 == 0
            println("PCFM step: $step/$n_steps")
        end

        τ = step * dt
        τ_next = τ + dt
        t_vec = fill(τ, n_samples) |> device

        x_input = prepare_input_fn(x, t_vec, nx, nt, n_samples, emb_channels)
        v, st = model_fn(x_input, ps, st)

        # Step 1: extrapolate to t = 1.
        x_1_hat = x .+ v .* (1.0f0 - τ)

        # Step 2: project onto the constraint manifold. `project` allocates and returns a
        # fresh array; for tight loops the mutating `project!` could be used with a
        # preallocated buffer, but allocation is cheap relative to the FNO forward pass.
        x_1 = project(solver, x_1_hat, constraint)

        # Step 3: re-interpolate between x_0 and the corrected x_1 at t + dt.
        x = x_0 .+ (x_1 .- x_0) .* τ_next
    end

    return x
end
