# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Summary

**MoireIVC.jl** is a Julia package for studying interacting valley Chern (IVC) states in moiré topological insulator systems, specifically focusing on transition metal dichalcogenides (TMDs) like MoTe₂. The package provides comprehensive tools for both Hartree-Fock mean-field calculations and exact diagonalization studies of valley-coherent states.

### Project Structure

- **`src/MoireIVC.jl`** - Main module definition
- **`src/methods/`** - Core computational methods
  - `ExactDiagonalization.jl` - General momentum-resolved ED framework
  - `HartreeFock.jl` - General Hartree-Fock self-consistent iteration methods
- **`src/content/`** - Physics-specific implementations
  - `basicthings.jl` - Core utilities (Landau level form factors, lattice functions)
  - `LL HF.jl` - Hartree-Fock calculations for Landau levels in TMD moiré systems
  - `LL ED.jl` - Exact diagonalization for moiré Dirac cone systems
  - `LL HF EDGE.jl` - Edge state calculations for moiré systems
  - `LL TDHF.jl` - Time-dependent Hartree-Fock methods
- **`src/plot/`** - Visualization utilities
- **`examples/`** - Jupyter notebooks demonstrating usage
- **`test/`** - Test suite for validation

### Completed Functionalities

#### 1. **Core Physics Engine**
- **Landau Level Physics**: Complete implementation of Landau level form factors and wavefunctions for TMD systems
- **Moiré Lattice**: Hexagonal lattice geometry with proper reciprocal lattice vectors
- **Interaction Models**: Coulomb interactions with gate screening and valley-dependent form factors

#### 2. **Hartree-Fock Framework (`LL HF.jl`)**
- **Self-Consistent Solver**: Advanced Hartree-Fock solver with dynamic mixing and convergence controls
- **Crystal Symmetries**: Built-in symmetry enforcement (C3 rotation, translation, parity, time-reversal)
- **Valley Physics**: Two-valley model with intervalley coherence and Berry curvature calculations
- **Material Parameters**: MoTe₂ system with realistic band parameters and screening

#### 3. **Exact Diagonalization (`LL ED.jl`)**
- **Momentum-Resolved ED**: Complete implementation for moiré Dirac cone systems
- **Many-Body States**: Efficient handling of many-body states with momentum conservation
- **Reduced Density Matrices**: One-body and subsystem reduced density matrices
- **Entanglement Analysis**: Entanglement entropy calculations for ground states

#### 4. **Advanced Features**
- **Edge States**: Edge calculation framework for studying boundary effects
- **Time-Dependent Methods**: TDHF implementation for dynamical response
- **Visualization**: Comprehensive plotting capabilities for band structures and phase diagrams

### Ongoing Work

Based on recent git commits and current development:

1. **Edge State Development**: Active work on `LL HF EDGE.jl` for edge state calculations
2. **Crystal Symmetry Analysis**: Recent focus on comparing energies of solutions with different crystal symmetries
3. **Code Refactoring**: Migration to general Hartree-Fock method code (`use general hartree-fock method code`)
4. **Performance Optimization**: Precompilation improvements and code organization

### Key Dependencies

- **Linear Algebra**: MKL, LinearAlgebra
- **Physics Libraries**: PhysicalUnits.jl (custom fork), ClassicalOrthogonalPolynomials
- **Numerical Methods**: KrylovKit, QuadGK, ExtendableSparse
- **Visualization**: CairoMakie
- **Utilities**: TensorOperations, Combinatorics

### Usage Examples

The package includes comprehensive Jupyter notebooks in `examples/`:
- `LLED.ipynb`: Exact diagonalization for moiré Dirac cones
- `LLHF.ipynb`: Hartree-Fock calculations for valley-polarized states

### Current Development Focus
- **General Exact Diagonalization method**: Formulate the general ED calculation with input k points and interactions
- **Edge State Physics**: Developing robust edge state calculations for moiré systems
- **Symmetry Breaking**: Analyzing spontaneous symmetry breaking in valley-polarized states
- **Performance**: Optimizing memory usage and computational efficiency for large systems

## Rules

Using Windows file system.
Disregard files in the "test" and "publication figures" folders unless they are specifically referred to.