"""
Example script for training Functional Flow Matching on 1D diffusion equation.

This script demonstrates:
1. Creating an FFM (Functional Flow Matching) model
2. Generating training data
3. Compiling functions with Reactant
4. Training the model
5. Generating samples
"""

using PCFM
using Reactant, Lux
using Plots

# Set random seed
using Random
Random.seed!(1234)

# Configuration
batch_size = 32
nx = 100          # Spatial resolution
nt = 100          # Temporal resolution
emb_channels = 32
n_epochs = 1000

# Data generation parameters
visc_range = (1.0f0, 5.0f0)
phi_range = (0.0f0, Float32(π))
t_range = (0.0f0, 1.0f0)

println("=" ^ 60)
println("Training Functional Flow Matching on 1D Diffusion Equation")
println("=" ^ 60)

# 1. Generate training data
println("\n[1/5] Generating training data...")
u_data = generate_diffusion_data(batch_size, nx, nt, visc_range, phi_range, t_range)
println("  Data shape: $(size(u_data))")

# 2. Create model
println("\n[2/5] Creating FFM (Functional Flow Matching) model...")
ffm = FFM(
    nx = nx,
    nt = nt,
    emb_channels = emb_channels,
    hidden_channels = 64,
    proj_channels = 256,
    n_layers = 4,
    modes = (32, 32),
    device = reactant_device()
);
println("  Model created successfully")

# 3. Compile functions (optional but recommended for speed)
println("\n[3/5] Compiling functions with Reactant...")
compiled_funcs = PCFM.compile_functions(ffm, batch_size)

# 4. Train model
println("\n[4/5] Training model for $n_epochs epochs...")
losses, tstate = train_ffm!(ffm, u_data; compiled_funcs, epochs = n_epochs, verbose = true)

println("\nFinal loss: $(losses[end])")

# 5. Generate samples
println("\n[5/5] Generating samples...")
n_samples = 32
# samples = sample_ffm(ffm, tstate, n_samples, 100; compiled_funcs, verbose = true)

samples = sample_pcfm(ffm, tstate, n_samples, 100; compiled_funcs, verbose = true)


println("\n" * "=" ^ 60)
println("Training Complete!")
println("=" ^ 60)

# Visualize results
println("\nPlotting results...")

# Plot training curve
p1 = plot(1:length(losses), losses,
    yscale = :log10,
    xlabel = "Epoch",
    ylabel = "Loss (log scale)",
    title = "Training Loss",
    legend = false,
    linewidth = 2)

# Plot samples
arr_data = Array(u_data)
arr_samples = Array(samples)

p_data = [heatmap(arr_data[:, :, 1, i],
              title = "Training Data $i",
              xlabel = "Time",
              ylabel = "Space",
              c = :viridis)
          for i in 1:min(2, batch_size)]

p_samples = [heatmap(arr_samples[:, :, 1, i],
                 title = "Generated Sample $i",
                 xlabel = "Time",
                 ylabel = "Space",
                 c = :viridis)
             for i in 1:2]

# Combine plots
p2 = plot(p_data..., layout = (1, length(p_data)), size = (800, 300))
p3 = plot(p_samples..., layout = (1, length(p_samples)), size = (800, 300))

display(p1)
display(p2)
display(p3)

println("\nDone! Check the plots above.")
