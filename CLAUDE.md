# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

windows file system uses \

## Project Overview

This is a Julia package for studying **Moire Interlayer Valley Coherence (MoireIVC)** - a physics research codebase focused on Hartree-Fock calculations for Landau levels in moiré systems with opposite magnetic fields. The package implements numerical methods for studying valley-coherent states in twisted bilayer systems.

## Architecture & Key Components

### Core Modules
- **Basics**: Fundamental physics utilities (Landau level form factors, Weierstrass functions, lattice calculations)
- **LLHF**: Hartree-Fock calculations for Landau levels - main computational engine
- **LLTDHF**: Time-dependent Hartree-Fock methods
- **LLED**: Exact diagonalization methods
- **Plotting**: Visualization utilities for results

### Key Data Structures
- `LLHFSysPara`: System parameters (material constants, Landau level indices)
- `LLHFNumPara`: Numerical parameters (grid sizes, convergence criteria)
- Uses PhysicalUnits.jl for unit handling throughout

### Computational Methods
1. **Hartree-Fock Solver**: Self-consistent field calculations for valley-coherent states
2. **Symmetry Operations**: C3 rotation, PT symmetry implementations
3. **Berry Curvature**: Topological property calculations
4. **Real-space Representations**: Pauli matrix operations in real space

## Development Commands

### Testing
```bash
# Run all tests
julia test/runtests.jl

# Run specific test files
julia test/testalpha.jl
julia test/testEDcoed.jl
julia test/testedge.jl
julia test/testTDHF.jl
julia test/testwavefunction.jl

# Interactive testing
julia --project
using MoireIVC
using Test
```

### Package Management
```bash
# Install dependencies
julia --project -e 'using Pkg; Pkg.instantiate()'

# Update dependencies
julia --project -e 'using Pkg; Pkg.update()'

# Add new packages
julia --project -e 'using Pkg; Pkg.add("PackageName")'
```

### Build & CI
The project uses GitHub Actions for CI/CD with Julia 1.11. Build process:
1. Dependencies installed via `julia-actions/cache`
2. Package built with `julia-actions/julia-buildpkg`
3. Tests run with `julia-actions/julia-runtest` (uses xvfb for GUI)

### Running Examples
```bash
# Start Julia with project
julia --project

# Load and use package
using MoireIVC
using MoireIVC.LLHF

# Initialize system
num_para = LLHF_init_with_lambda(0.3; N1=1, N2=1)
```

### Jupyter Notebooks
Interactive examples located in `examples/`:
- `LLED.ipynb`: Landau level exact diagonalization
- `LLHF.ipynb`: Hartree-Fock calculations

## Key Workflows

### Hartree-Fock Calculation Pipeline
1. Initialize system: `LLHF_init_with_lambda()` or `LLHF_init_with_alpha()`
2. Modify parameters: `LLHF_change_lambda!()` or `LLHF_change_alpha!()`
3. Set symmetries: Define symmetry operations (Rot3, PT)
4. Solve: `LLHF_solve()` with convergence parameters
5. Analyze: Energy calculations, berry curvature, real-space properties

### Parameter Sweeps
Common pattern seen in test files:
- Sweep lambda parameter (interaction strength)
- Sweep alpha parameter (twist angle)
- Monitor convergence and energy per area

## Files to Know
- `src/content/basicthings.jl`: Core physics utilities
- `src/content/LL HF.jl`: Main Hartree-Fock implementation
- `src/methods/`: Algorithmic implementations
- `examples/`: Interactive notebooks for exploration
- `publication figures/`: Generated results and analysis