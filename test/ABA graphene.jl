using LinearAlgebra # For matrix operations and eigenvalues
using CairoMakie    # For high-quality plotting

# Define physical constants (using typical values)
const vF_hbar = 0.6582 # eV * nm (approximate, for k in nm^-1 and energy in eV)
const gamma1 = 0.38 # eV (typical value for interlayer hopping)

"""
    ABAHamiltonian(kx, ky, vF_hbar, gamma1)

Generates the 6x6 Hamiltonian matrix for ABA-stacked trilayer graphene
at a given momentum (kx, ky).

Arguments:
- `kx`: Momentum in the x-direction.
- `ky`: Momentum in the y-direction.
- `vF_hbar`: Product of Fermi velocity and reduced Planck constant (e.g., in eV*nm).
- `gamma1`: Interlayer hopping parameter (e.g., in eV).

Returns:
- A 6x6 ComplexF64 matrix representing the Hamiltonian.
"""
function ABAHamiltonian(kx::Real, ky::Real, vF_hbar::Real, gamma1::Real)
    k_plus = kx + im * ky
    k_minus = kx - im * ky

    H = zeros(ComplexF64, 6, 6)

    # Intra-layer hopping (diagonal blocks)
    # Layer 1 (psi_A1, psi_B1)
    H[1, 2] = vF_hbar * k_plus
    H[2, 1] = vF_hbar * k_minus

    # Layer 2 (psi_A2, psi_B2)
    H[3, 4] = vF_hbar * k_plus
    H[4, 3] = vF_hbar * k_minus

    # Layer 3 (psi_A3, psi_B3)
    H[5, 6] = vF_hbar * k_plus
    H[6, 5] = vF_hbar * k_minus

    # Inter-layer hopping (gamma1)
    # A1 <-> B2
    H[1, 4] = gamma1
    H[4, 1] = gamma1

    # B2 <-> A3
    H[4, 5] = gamma1
    H[5, 4] = gamma1

    return H
end

# --- Plotting Energy Bands with CairoMakie ---

# Define the range for kx (e.g., around the Dirac point)
kx_values = range(-0.5, stop=0.5, length=200) # k in nm^-1

# Fix ky to 0
fixed_ky = 0.0

# Initialize an array to store eigenvalues for each kx
num_bands = 6
eigenvalues = zeros(length(kx_values), num_bands);
eigenvectors = zeros(length(kx_values), num_bands, num_bands);

# Loop through kx values, calculate Hamiltonian, and find eigenvalues
for (i, kx) in enumerate(kx_values)
    H = ABAHamiltonian(kx, fixed_ky, vF_hbar, gamma1)
    eigenvalues[i, :], eigenvectors[i, :, :] = eigen(H)
end

begin
    # Create a Figure and an Axis object for plotting
    f = Figure(size = (800, 600));
    ax = Axis(f[1, 1],
        xlabel = "kx (nm⁻¹)",
        ylabel = "Energy (eV)",
        subtitle = "ABA Trilayer Graphene Energy Bands (ky=0)",
    )

    # Plot each band
    for band_idx in 1:num_bands
        lines!(ax, kx_values, eigenvalues[:, band_idx], label="Band $(band_idx)",
            linewidth=2) # CairoMakie does not have a 'marker' keyword for lines, use scatter for markers
    end

    # Add a legend
    axislegend(ax, position = :rt) # Position can be :lt, :lc, :lb, :rt, :rc, :rb, :ct, :cc, :cb

    # Display the plot
    f
end

begin
    band_number = 4

    f = Figure(size = (800, 600));
    ax = Axis(f[1, 1],
        xlabel = "kx (nm⁻¹)",
        ylabelvisible = false,
        yticks=(1:6, ["1A", "1B", "2A", "2B", "3A", "3B"]),
        subtitle = "ABA Trilayer Graphene band #$(band_number) (ky=0)"
    )

    hm = heatmap!(ax, kx_values, 1:num_bands, abs.(eigenvectors[:, :, band_number]);
        colorrange=(0,1), 
        colormap=range(Makie.Colors.colorant"white", stop=Makie.Colors.colorant"#ec2f41", length=15),
    )

    # Display the plot
    f
end