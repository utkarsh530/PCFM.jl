"""
    generate_diffusion_data(n_samples, nx, nt, visc_range, phi_range, t_range)

Generate 1D diffusion dataset (space-time).

# Arguments

  - `n_samples`: Number of samples to generate
  - `nx`: Number of spatial points
  - `nt`: Number of temporal points
  - `visc_range`: Tuple of (min, max) viscosity values
  - `phi_range`: Tuple of (min, max) phase shift values
  - `t_range`: Tuple of (start, end) time range

# Returns

  - `u_data`: Array of shape (nx, nt, 1, n_samples) containing the diffusion solutions

# Example

```julia
u_data = generate_diffusion_data(
    32, 100, 100, (1.0f0, 5.0f0), (0.0f0, Float32(π)), (0.0f0, 1.0f0))
```

The data follows the analytical solution: u(x,t) = sin(x + φ) * exp(-t * ν)
"""
function generate_diffusion_data(n_samples, nx, nt, visc_range, phi_range, t_range)
    xs = range(0.0f0, 2.0f0 * Float32(π), length = nx+1)[1:(end - 1)]
    ts = range(t_range[1], t_range[2], length = nt)

    # Julia format: (nx, nt, 1, n_samples) = (H, W, C, B)
    u_data = zeros(Float32, nx, nt, 1, n_samples)

    for i in 1:n_samples
        v = visc_range[1] + rand(Float32) * (visc_range[2] - visc_range[1])
        phi = phi_range[1] + rand(Float32) * (phi_range[2] - phi_range[1])

        # u(x,t) = sin(x + phi) * exp(-t * v)
        for (ti, t) in enumerate(ts)
            u_data[:, ti, 1, i] .= sin.(xs .+ phi) .* exp(-t * v)
        end
    end

    return u_data
end
