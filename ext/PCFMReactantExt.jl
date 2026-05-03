
"""

    PCFMReactantExt



Weak-dependency extension loaded when `Reactant` is present. Restores the

real `reactant_device()` and the JIT-compile path for the FFM model. Without

this extension, PCFM still works end-to-end on CPU arrays — `reactant_device()`

returns `identity`, and `compile_functions` raises a clear error.

"""

module PCFMReactantExt



using PCFM

using Reactant



# Override the stubs in the core module. `PCFM.reactant_device` and

# `PCFM._reactant_compile` are defined as stubs in src/PCFM.jl so that

# the names exist even without this extension; here we replace their

# behaviour with the real Reactant calls.



PCFM.reactant_device() = Reactant.reactant_device()



# The stub takes a thunk `f` and would normally call `Reactant.@compile f()`.

# We can't @compile a thunk directly (the macro expects a call expression),

# so we evaluate the thunk inside a generated wrapper. In practice the

# call sites pass `() -> ffm.model(x_test, ffm.ps, ...)` so we just compile

# whatever the thunk does.

function PCFM._reactant_compile(f; kwargs...)

    return Reactant.@compile f()

end



end # module

