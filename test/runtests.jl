using Test
using PCFM
using Random

@testset "PCFM.jl Tests" begin
    
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
            nx=50,
            nt=50,
            emb_channels=16,
            hidden_channels=32,
            proj_channels=128,
            n_layers=2
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
        
        # Test boundary conditions
        t_zero = zeros(Float32, n_samples)
        x_t_0 = interpolate_flow(t_zero, x_0, data, n_samples)
        @test x_t_0 ≈ x_0
        
        t_one = ones(Float32, n_samples)
        x_t_1 = interpolate_flow(t_one, x_0, data, n_samples)
        @test x_t_1 ≈ data
    end
    
    @testset "Training (small)" begin
        Random.seed!(1234)
        
        # Small model for fast testing
        n_samples = 4
        nx, nt = 20, 20
        
        ffm = FFM(
            nx=nx,
            nt=nt,
            emb_channels=8,
            hidden_channels=16,
            proj_channels=32,
            n_layers=1
        )
        
        data = generate_diffusion_data(n_samples, nx, nt, (1.0f0, 2.0f0), (0.0f0, 1.0f0), (0.0f0, 1.0f0))
        
        losses, tstate = train_ffm!(ffm, data; epochs=5, verbose=false)
        
        @test length(losses) == 5
        @test all(isfinite.(losses))
        @test losses[end] < losses[1]  # Loss should decrease
    end
    
    @testset "Sampling (small)" begin
        Random.seed!(1234)
        
        n_samples = 2
        nx, nt = 20, 20
        
        ffm = FFM(
            nx=nx,
            nt=nt,
            emb_channels=8,
            hidden_channels=16,
            proj_channels=32,
            n_layers=1
        )
        
        data = generate_diffusion_data(4, nx, nt, (1.0f0, 2.0f0), (0.0f0, 1.0f0), (0.0f0, 1.0f0))
        losses, tstate = train_ffm!(ffm, data; epochs=2, verbose=false)
        
        samples = sample_ffm(ffm, tstate, n_samples, 10; verbose=false)
        
        @test size(samples) == (nx, nt, 1, n_samples)
        @test eltype(samples) == Float32
        @test all(isfinite.(samples))
    end
    
end
