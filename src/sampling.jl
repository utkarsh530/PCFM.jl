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

"""
    sample_pcfm(ffm::FFM, tstate, n_samples, n_steps;
                use_compiled=false, compiled_funcs=nothing, verbose=true)

Generate physics-constrained samples using PCFM algorithm.
Fixed initial condition: u(x,0) = sin(x + π/4)

# Arguments
  - `ffm`: FFM model
  - `tstate`: Training state
  - `n_samples`: Number of samples to generate
  - `n_steps`: Number of integration steps
  - `use_compiled`: Whether to use compiled functions
  - `compiled_funcs`: Compiled functions
  - `verbose`: Print progress

# Returns
  - Generated samples satisfying the initial condition constraint
"""
function sample_pcfm(ffm::FFM, tstate, n_samples, n_steps;
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

    # Fixed initial condition: u(x,0) = sin(x + π/4)
    x_grid = range(0, 2π, length=nx)
    u_0_ic = sin.(x_grid .+ π/4)
    u_0_ic = reshape(u_0_ic, nx, 1, 1, 1)
    # Broadcast to all samples
    u_0_ic = repeat(u_0_ic, 1, 1, 1, n_samples) |> device

    # Start from Gaussian noise
    x_0 = randn(Float32, nx, nt, 1, n_samples) |> device
    x = copy(x_0)

    dt = 1.0f0 / n_steps

    # Euler integration from t=0 to t=1
    for step in 0:(n_steps - 1)
        if verbose && step % 10 == 0
            println("PCFM step: $step/$n_steps")
        end

        τ = step * dt
        τ_next = τ + dt
        t_vec = fill(τ, n_samples) |> device

        # Prepare input with embeddings
        x_input = prepare_input_fn(x, t_vec, nx, nt, n_samples, emb_channels)

        # Predict velocity field
        v, st = model_fn(x_input, ps, st)

        # Step 1: Extrapolate to t=1
        x_1 = x .+ v .* (1.0f0 - τ)

        # Step 2: Apply constraint - fix initial condition
        @. x_1[:, 1:1, :, :] = u_0_ic

        # Step 3: Interpolate between x_0 and x_1 (corrected) at time t+dt
        x = x_0 .+ (x_1 .- x_0) .* τ_next
    end

    return x
end
