"""
    FFM

Structure holding the Fourier Neural Operator model for Functional Flow Matching.

# Fields

  - `model`: The FNO model
  - `ps`: Parameters
  - `st`: States
  - `config`: Configuration dictionary

# Configuration

  - `nx`: Spatial resolution
  - `nt`: Temporal resolution
  - `emb_channels`: Number of time embedding channels
  - `hidden_channels`: Number of hidden channels in FNO
  - `proj_channels`: Number of projection channels
  - `n_layers`: Number of FNO layers
  - `modes`: Fourier modes tuple
"""
struct FFM{M, P, S}
    model::M
    ps::P
    st::S
    config::Dict{Symbol, Any}
end

"""
    FFM(; nx=100, nt=100, emb_channels=32, hidden_channels=64,
        proj_channels=256, n_layers=4, modes=(32, 32),
        device=reactant_device(), rng=Random.default_rng())

Create a Functional Flow Matching model with FNO backbone.

# Example

```julia
model = FFM(nx = 100, nt = 100, emb_channels = 32)
```
"""
function FFM(;
        nx = 100,
        nt = 100,
        emb_channels = 32,
        hidden_channels = 64,
        proj_channels = 256,
        n_layers = 4,
        modes = (32, 32),
        device = reactant_device(),
        rng = Random.default_rng()
)
    in_channels = 1 + emb_channels + 2  # u + time_emb + pos_x + pos_t

    fno = FourierNeuralOperator(
        modes,
        in_channels,
        1,  # output channels
        hidden_channels;
        num_layers = n_layers,
        lifting_channel_ratio = proj_channels ÷ hidden_channels,
        projection_channel_ratio = proj_channels,
        activation = gelu,
        fno_skip = :linear,
        channel_mlp_skip = :soft_gating,
        use_channel_mlp = true,
        channel_mlp_expansion = 1.0,
        positional_embedding = :none,
        stabilizer = tanh
    )

    ps, st = Lux.setup(rng, fno) |> device

    config = Dict{Symbol, Any}(
        :nx => nx,
        :nt => nt,
        :emb_channels => emb_channels,
        :hidden_channels => hidden_channels,
        :proj_channels => proj_channels,
        :n_layers => n_layers,
        :modes => modes,
        :device => device
    )

    return FFM(fno, ps, st, config)
end

"""
    compile_model(ffm::FFM)

Compile the FNO model for faster execution with Reactant.

Returns compiled model function.
"""
function compile_model(ffm::FFM)
    nx = ffm.config[:nx]
    nt = ffm.config[:nt]
    emb_channels = ffm.config[:emb_channels]
    device = ffm.config[:device]

    # Test input for compilation
    x_test = randn(Float32, nx, nt, 1 + emb_channels + 2, 32) |> device

    model_compiled = Reactant.@compile ffm.model(x_test, ffm.ps, Lux.testmode(ffm.st))

    return model_compiled
end

"""
    prepare_input(x_t, t, nx, nt, n_samples, emb_dim; max_positions=2000)

Prepare input tensor for FNO by concatenating state, time embedding, and position embeddings.

# Arguments

  - `x_t`: Current state (nx, nt, 1, n_samples)
  - `t`: Time values (n_samples,)
  - `nx`, `nt`: Spatial dimensions
  - `n_samples`: Batch size
  - `emb_dim`: Time embedding dimension
  - `max_positions`: Maximum position for time embedding scaling

# Returns

  - Input tensor of shape (nx, nt, 1+emb_dim+2, n_samples)
"""
function prepare_input(x_t, t, nx, nt, n_samples, emb_dim; max_positions = 2000)
    u_channel = x_t

    # Time embedding (sinusoidal)
    timesteps = t .* Float32(max_positions)
    half_dim = emb_dim ÷ 2

    emb_scale = Float32(log(max_positions)) / Float32(half_dim - 1)
    emb_base = exp.(Float32.(-collect(0:(half_dim - 1)) .* emb_scale))
    t_emb = timesteps * emb_base'
    t_emb = hcat(sin.(t_emb), cos.(t_emb))

    # Reshape and broadcast to spatial dimensions
    t_emb = permutedims(t_emb, (2, 1))
    t_emb = reshape(t_emb, 1, 1, emb_dim, n_samples)
    t_emb = repeat(t_emb, nx, nt, 1, 1)

    # Position embeddings (normalized coordinates)
    pos_x = range(0.0f0, 1.0f0, length = nx)
    pos_t = range(0.0f0, 1.0f0, length = nt)
    pos_x_grid = repeat(reshape(collect(pos_x), nx, 1, 1, 1), 1, nt, 1, n_samples)
    pos_t_grid = repeat(reshape(collect(pos_t), 1, nt, 1, 1), nx, 1, 1, n_samples)

    # Concatenate all channels
    x_input = cat(u_channel, t_emb, pos_x_grid, pos_t_grid; dims = 3)

    return x_input
end

"""
    interpolate_flow(t, x_0, data, n_samples)

Linear interpolation between noise and data for flow matching.

x_t = (1-t)*x_0 + t*x_1

# Arguments

  - `t`: Time values (n_samples,)
  - `x_0`: Noise/initial state
  - `data`: Target data
  - `n_samples`: Batch size

# Returns

  - Interpolated state x_t
"""
function interpolate_flow(t, x_0, data, n_samples)
    t_expanded = reshape(t, 1, 1, 1, n_samples)
    x_t = (1 .- t_expanded) .* x_0 .+ t_expanded .* data
    return x_t
end

"""
    compile_functions(ffm::FFM, batch_size::Int)

Compile all helper functions (model, interpolation, input preparation) with Reactant.

Returns a NamedTuple with compiled functions.
"""
function compile_functions(ffm::FFM, batch_size::Int)
    nx = ffm.config[:nx]
    nt = ffm.config[:nt]
    emb_channels = ffm.config[:emb_channels]
    device = ffm.config[:device]

    # Test data
    x_test = randn(Float32, nx, nt, 1 + emb_channels + 2, batch_size) |> device
    t_test = rand(Float32, batch_size) |> device
    x_0_test = randn(Float32, nx, nt, 1, batch_size) |> device
    data_test = randn(Float32, nx, nt, 1, batch_size) |> device

    # Compile model
    model_compiled = Reactant.@compile ffm.model(x_test, ffm.ps, Lux.testmode(ffm.st))

    # Compile interpolation
    interpolate_compiled = Reactant.@compile interpolate_flow(t_test, x_0_test, data_test, batch_size)

    # Compile input preparation
    x_t_test = interpolate_flow(t_test, x_0_test, data_test, batch_size)
    prepare_input_compiled = Reactant.@compile prepare_input(
        x_t_test, t_test, nx, nt, batch_size, emb_channels)

    return (
        model = model_compiled,
        interpolate = interpolate_compiled,
        prepare_input = prepare_input_compiled
    )
end
