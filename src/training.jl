"""
    train_ffm!(ffm::FFM, data; epochs=1000, lr=0.001f0,
               use_compiled=false, compiled_funcs=nothing, verbose=true)

Train the Functional Flow Matching model using the given data.

# Arguments
- `ffm`: FFM model
- `data`: Training data of shape (nx, nt, 1, n_samples)
- `epochs`: Number of training epochs
- `lr`: Learning rate
- `use_compiled`: Whether to use compiled functions
- `compiled_funcs`: Compiled functions from `compile_functions`
- `verbose`: Print progress

# Returns
- `losses`: Array of training losses
- `tstate`: Final training state

# Example
```julia
ffm = FFM()
data = generate_diffusion_data(32, 100, 100, (1.0f0, 5.0f0), (0.0f0, Float32(π)), (0.0f0, 1.0f0))
losses, tstate = train_ffm!(ffm, data; epochs=1000)
```
"""
function train_ffm!(ffm::FFM, data;
                    epochs=1000,
                    lr=0.001f0,
                    use_compiled=true,
                    compiled_funcs=nothing,
                    verbose=true)

    nx = ffm.config[:nx]
    nt = ffm.config[:nt]
    emb_channels = ffm.config[:emb_channels]
    device = ffm.config[:device]

    losses = Float32[]
    tstate = Training.TrainState(ffm.model, ffm.ps, ffm.st, Adam(lr))

    _, _, _, n_samples = size(data)
    data = data |> device

    # Use compiled or regular functions
    if use_compiled && compiled_funcs !== nothing
        interpolate_fn = compiled_funcs.interpolate
        prepare_input_fn = compiled_funcs.prepare_input
    else
        interpolate_fn = interpolate_flow
        prepare_input_fn = prepare_input
    end

    for epoch in 1:epochs
        # Sample random time t ∈ [0, 1]
        t = rand(Float32, n_samples) |> device

        # Sample Gaussian noise
        x_0 = randn(Float32, nx, nt, 1, n_samples) |> device

        # Interpolation: x_t = (1-t)*x_0 + t*data
        x_t = interpolate_fn(t, x_0, data, n_samples)

        # Target velocity field: v = data - x_0
        v_target = data .- x_0

        # Prepare input with embeddings
        x_input = prepare_input_fn(x_t, t, nx, nt, n_samples, emb_channels)

        # Training step
        (_, loss, _, tstate) = Training.single_train_step!(
            AutoEnzyme(), MSELoss(), (x_input, v_target), tstate;
            return_gradients=Val(false)
        )

        push!(losses, Float32(loss))

        if verbose && (epoch % 100 == 0 || epoch == 1)
            println("Epoch $epoch: Loss = $(Float32(loss))")
        end
    end

    return losses, tstate
end
