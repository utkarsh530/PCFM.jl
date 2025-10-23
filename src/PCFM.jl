module PCFM

using NeuralOperators
using Lux
using Random
using Optimisers
using Reactant

# Include submodules
include("./data.jl")
include("./model.jl")
include("./training.jl")
include("./sampling.jl")

# Export main functions
export FFM
export prepare_input, interpolate_flow
export train_ffm!, sample_ffm
export generate_diffusion_data


end # module PCFM
