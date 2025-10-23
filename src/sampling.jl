"""
    sample_ffm(ffm::FFM, tstate, n_samples, n_steps;
               use_compiled=false, compiled_funcs=nothing, verbose=true)

Generate samples from the trained Functional Flow Matching model using Euler integration.

# Arguments

  - `ffm`: FFM model
  - `tstate`: Training state (or use `ffm.ps` and `ffm.st` directly)
  - `n_samples`: Number of samples to generate
  - `n_steps`: Number of Euler integration steps
  - `use_compiled`: Whether to use compiled functions
  - `compiled_funcs`: Compiled functions from `compile_functions`
  - `verbose`: Print progress

# Returns

  - Generated samples of shape (nx, nt, 1, n_samples)

# Example

```julia
samples = sample_ffm(ffm, tstate, 32, 100)
```
"""
function sample_ffm(ffm::FFM, tstate, n_samples, n_steps;
        use_compiled = true,
        compiled_funcs = nothing,
        verbose = true)
    nx = ffm.config[:nx]
    nt = ffm.config[:nt]
    emb_channels = ffm.config[:emb_channels]
    device = ffm.config[:device]

    # Extract parameters and states
    if hasfield(typeof(tstate), :parameters)
        ps = tstate.parameters
        st = tstate.states
    else
        ps = tstate[1]
        st = tstate[2]
    end

    # Use compiled or regular functions
    if use_compiled && compiled_funcs !== nothing
        model_fn = compiled_funcs.model
        prepare_input_fn = compiled_funcs.prepare_input
    else
        model_fn = ffm.model
        prepare_input_fn = prepare_input
    end

    # Start from Gaussian noise
    x = randn(Float32, nx, nt, 1, n_samples) |> device
    dt = 1.0f0 / n_steps

    # Euler integration from t=0 to t=1
    for step in 0:(n_steps - 1)
        if verbose && step % 10 == 0
            println("Sampling step: $step/$n_steps")
        end

        t_scalar = step * dt
        t_vec = fill(t_scalar, n_samples) |> device

        # Prepare input with embeddings
        x_input = prepare_input_fn(x, t_vec, nx, nt, n_samples, emb_channels)

        # Predict velocity field
        if use_compiled
            v, st = model_fn(x_input, ps, st)
        else
            v, st = model_fn(x_input, ps, st)
        end

        # Update state: x ← x + v * dt
        x = x .+ v .* dt
    end

    return x
end
