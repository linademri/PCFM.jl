#=
train_and_save.jl

Train a small FFM on the 1D diffusion benchmark and persist the training state + model
config to disk using Julia's Serialization. The resulting file is consumed by
`benchmark_sample_pcfm.jl` so benchmarks can be run many times without retraining.

Usage:
    julia --project=. benchmarks/train_and_save.jl [output_path]

Env vars:
    PCFM_BENCH_NX=100
    PCFM_BENCH_NT=100
    PCFM_BENCH_EPOCHS=500      # reduce for quick tests
    PCFM_BENCH_HIDDEN=64
    PCFM_BENCH_MODES=32
    PCFM_BENCH_TRAIN_BS=32
=#

using PCFM
using Reactant
using Serialization
using Random

const NX         = parse(Int, get(ENV, "PCFM_BENCH_NX", "100"))
const NT         = parse(Int, get(ENV, "PCFM_BENCH_NT", "100"))
const EPOCHS     = parse(Int, get(ENV, "PCFM_BENCH_EPOCHS", "500"))
const HIDDEN     = parse(Int, get(ENV, "PCFM_BENCH_HIDDEN", "64"))
const MODES      = parse(Int, get(ENV, "PCFM_BENCH_MODES", "32"))
const TRAIN_BS   = parse(Int, get(ENV, "PCFM_BENCH_TRAIN_BS", "32"))
const SEED       = parse(Int, get(ENV, "PCFM_BENCH_SEED", "20260420"))

const OUT_PATH = length(ARGS) >= 1 ? ARGS[1] : "ffm_checkpoint.jls"

Random.seed!(SEED)

println("Training FFM for benchmarking")
println("  nx = $NX, nt = $NT, epochs = $EPOCHS, train batch = $TRAIN_BS")
println("  hidden = $HIDDEN, modes = $MODES")
println("  output: $OUT_PATH")

# 1. Data.
u_data = generate_diffusion_data(
    TRAIN_BS, NX, NT,
    (1.0f0, 5.0f0), (0.0f0, Float32(π)), (0.0f0, 1.0f0),
)

# 2. Model.
ffm = FFM(
    nx = NX, nt = NT,
    emb_channels = 32,
    hidden_channels = HIDDEN,
    proj_channels = 256,
    n_layers = 4,
    modes = (MODES, MODES),
    device = reactant_device(),
)

# 3. Compile.
compiled_funcs = PCFM.compile_functions(ffm, TRAIN_BS)

# 4. Train.
losses, tstate = train_ffm!(ffm, u_data; compiled_funcs, epochs = EPOCHS, verbose = true)

# 5. Save. We serialize the FFM config (a Dict{Symbol,Any}, easy to serialize) and the
# extracted (ps, st) rather than the whole FFM struct, because the FNO model object contains
# closures that can be finicky to deserialize across Julia/Lux versions. On load, we
# reconstruct FFM with the same config and inject the saved ps/st.
if hasfield(typeof(tstate), :parameters)
    ps_saved = tstate.parameters
    st_saved = tstate.states
else
    ps_saved = tstate[1]
    st_saved = tstate[2]
end

# Move to CPU before serializing — device arrays are not portable across sessions in
# general, and for benchmarks we move back to device on load anyway.
ps_cpu = ps_saved  # Lux parameters are named tuples; rely on Reactant's host-copy path
st_cpu = st_saved

ckpt = (
    config = ffm.config,
    ps = ps_cpu,
    st = st_cpu,
    losses = losses,
    final_loss = losses[end],
)

serialize(OUT_PATH, ckpt)
println("Saved checkpoint: $OUT_PATH  (final loss = $(losses[end]))")
