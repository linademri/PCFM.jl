# Preserved from the pre-0.2 test suite. These tests exercise model creation, training,
# and unconstrained sampling. They were the entire test suite before the projection
# refactor; keeping them intact verifies that the refactor is non-breaking.

@testset "Legacy tests" begin
    @testset "Data Generation" begin
        Random.seed!(1234)

        n_samples = 8
        nx, nt = 50, 50
        visc_range = (1.0f0, 5.0f0)
        phi_range = (0.0f0, Float32(π))
        t_range = (0.0f0, 1.0f0)

        data = generate_diffusion_data(n_samples, nx, nt, visc_range, phi_range, t_range)

        @test size(data) == (nx, nt, 1, n_samples)
        @test eltype(data) == Float32
        @test all(isfinite.(data))
    end

    @testset "Model Creation" begin
        Random.seed!(1234)

        ffm = FFM(
            nx = 50,
            nt = 50,
            emb_channels = 16,
            hidden_channels = 32,
            proj_channels = 128,
            n_layers = 2,
        )

        @test ffm.config[:nx] == 50
        @test ffm.config[:nt] == 50
        @test ffm.config[:emb_channels] == 16
        @test ffm.config[:hidden_channels] == 32
    end

    @testset "Input Preparation" begin
        Random.seed!(1234)

        nx, nt = 50, 50
        n_samples = 4
        emb_dim = 16

        x_t = randn(Float32, nx, nt, 1, n_samples)
        t = rand(Float32, n_samples)

        x_input = prepare_input(x_t, t, nx, nt, n_samples, emb_dim)

        @test size(x_input) == (nx, nt, 1 + emb_dim + 2, n_samples)
        @test eltype(x_input) == Float32
    end

    @testset "Interpolation" begin
        Random.seed!(1234)

        n_samples = 4
        nx, nt = 50, 50

        t = rand(Float32, n_samples)
        x_0 = randn(Float32, nx, nt, 1, n_samples)
        data = randn(Float32, nx, nt, 1, n_samples)

        x_t = interpolate_flow(t, x_0, data, n_samples)

        @test size(x_t) == size(x_0)
        @test eltype(x_t) == Float32

        t_zero = zeros(Float32, n_samples)
        @test interpolate_flow(t_zero, x_0, data, n_samples) ≈ x_0

        t_one = ones(Float32, n_samples)
        @test interpolate_flow(t_one, x_0, data, n_samples) ≈ data
    end

    @testset "Training (small)" begin
        Random.seed!(1234)

        n_samples = 4
        nx, nt = 20, 20

        ffm = FFM(
            nx = nx, nt = nt,
            emb_channels = 8, hidden_channels = 16, proj_channels = 32, n_layers = 1,
            modes = (8, 8),  # must be ≤ ⌈nx/2⌉+1 = 11 and ≤ nt = 20; keep small for tiny test model
        )

        data = generate_diffusion_data(
            n_samples, nx, nt, (1.0f0, 2.0f0), (0.0f0, 1.0f0), (0.0f0, 1.0f0),
        )

        losses, tstate = train_ffm!(ffm, data; epochs = 5, verbose = false)

        @test length(losses) == 5
        @test all(isfinite.(losses))
        @test losses[end] < losses[1]
    end

    @testset "Sampling (small)" begin
        Random.seed!(1234)

        n_samples = 2
        nx, nt = 20, 20

        ffm = FFM(
            nx = nx, nt = nt,
            emb_channels = 8, hidden_channels = 16, proj_channels = 32, n_layers = 1,
            modes = (8, 8),  # see note above
        )

        data = generate_diffusion_data(
            4, nx, nt, (1.0f0, 2.0f0), (0.0f0, 1.0f0), (0.0f0, 1.0f0),
        )
        # Compile for the training batch size (4) first — train_ffm! doesn't cache compiled
        # functions, and the uncompiled FFM forward pass fails on Reactant device arrays
        # (NNlib conv falls through to a CPU pointer path that can't handle ConcretePJRTArray).
        compiled_train = PCFM.compile_functions(ffm, 4)
        losses, tstate = train_ffm!(ffm, data; compiled_funcs = compiled_train, epochs = 2, verbose = false)

        # Separate compile for the sampling batch size (n_samples = 2).
        compiled_sample = PCFM.compile_functions(ffm, n_samples)
        samples = sample_ffm(ffm, tstate, n_samples, 10; compiled_funcs = compiled_sample, verbose = false)

        @test size(samples) == (nx, nt, 1, n_samples)
        @test eltype(samples) == Float32
        @test all(isfinite.(samples))
    end
end
