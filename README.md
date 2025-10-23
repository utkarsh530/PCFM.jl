# PCFM.jl

**Physics-Constrained Flow Matching**: Sampling Generative Models with Hard Constraints

[![arXiv](https://img.shields.io/badge/arXiv-2506.04171-b31b1b.svg)](https://arxiv.org/abs/2506.04171)

A Julia implementation for the NeurIPS 2025 Paper: Physics-Constrained Flow Matching: Sampling Generative Models with Hard Constraints

> **Note**: This package is under active development. More features and capabilities will be added progressively.


## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/utkarsh530/PCFM.jl")
```

Or in development mode:
```julia
Pkg.develop(path="/path/to/PCFM")
```

## Usage

For a complete working example, see [`examples/train_diffusion.jl`](examples/train_diffusion.jl).

To run the example:
```bash
julia --project examples/train_diffusion.jl
```

## Citation

If you use this package, please cite:

```bibtex
@article{utkarsh2025physics,
  title={Physics-Constrained Flow Matching: Sampling Generative Models with Hard Constraints},
  author={Utkarsh, Utkarsh and Cai, Pengfei and Edelman, Alan and Gomez-Bombarelli, Rafael and Rackauckas, Christopher Vincent},
  journal={arXiv preprint arXiv:2506.04171},
  year={2025}
}
```

## License

MIT License - see LICENSE file for details.
