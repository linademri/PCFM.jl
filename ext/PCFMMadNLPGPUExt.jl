"""
    PCFMMadNLPGPUExt

GPU variant of `PCFMMadNLPExt`. Loaded when `ExaModels`, `MadNLP`, and `MadNLPGPU` are all
present in the session. Provides a convenience constructor `MadNLPGPUSolver(; ...)` that
returns a `MadNLPSolver` pre-configured with `CUDABackend()` and MadNLP's GPU linear
solver (`CUDSSSolver`).

```julia
using PCFM, ExaModels, MadNLP, MadNLPGPU, CUDA
solver = MadNLPGPUSolver(tol = 1e-8)
Z = project(solver, Ẑ, constraint)
```

Everything else (constraint dispatch, ExaCore construction, MadNLP call) is inherited from
`PCFMMadNLPExt` — this extension only supplies the backend and linear solver.
"""
module PCFMMadNLPGPUExt

using PCFM
using ExaModels
using MadNLP
using MadNLPGPU
using CUDA

"""
    MadNLPGPUSolver(; kwargs...)

Convenience constructor returning a `MadNLPSolver` configured for GPU execution. All
`MadNLPSolver` kwargs are forwarded. Requires CUDA + MadNLPGPU to be loaded.
"""
function PCFM.MadNLPGPUSolver(;
    tol = 1e-8,
    max_iter = 200,
    print_level = MadNLP.ERROR,
)
    return PCFM.MadNLPSolver(
        tol = tol,
        max_iter = max_iter,
        print_level = print_level,
        backend = CUDABackend(),
        linear_solver = MadNLPGPU.CUDSSSolver,
    )
end

end # module
