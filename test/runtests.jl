using Test
using PCFM
using Random

@testset "PCFM.jl" begin
    # Existing tests preserved from pre-0.2 (model creation, training, sampling of the FFM).
    include("legacy_tests.jl")
    # Projection + constraint tests (CPU, pure Julia, always run).
    include("projection_tests.jl")
    # MadNLP backend tests — gated on ExaModels + MadNLP availability inside the file.
    include("madnlp_tests.jl")
end
