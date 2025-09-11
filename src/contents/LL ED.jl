"""
Do Exact Diagonalization for two regions in BZ.
The two regions contains two Dirac Points with large momentum kD1, kD2.
Small momentum is the momentum relative to Dirac Points (DP)
Use Landau level quasi-Bloch wavefunctions

Import system parameters from LLHF.
Only work for C3 solutions where the Dirac points are κ and κ′

Input numerical parameter includes:
Some k points around the two Dirac points,
One-body Hamiltonian determined by outside mean field.
Valley number is not a good quantum number.
Without Umklapp, electron number in each DP region is conserved.

Calculate the ED within the block specified with:
electron number in DC and total small momentum.

"""
module LLED

public LLEDPara
export LLED_init
# export LLED_mbslist_twoDiracCone, LLED_block_bysmallmomentum, LLED_interactionlist
# export LLED_Block_Hamiltonian, LLED_solve
# public reduced_density_matrix, one_body_reduced_density_matrix, entanglement_entropy 




using MKL, LinearAlgebra
using PhysicalUnits
using MomentumED
using MoireIVC.Basics: laguerrel, ql_cross, my_searchsortedfirst
using MoireIVC.LLHF: LLHFSysPara, LLHFNumPara

# initiation
LLEDPara = EDPara

# Define the form factor for Coulomb interaction in Landau level
# This is the Fourier transform of the projected Coulomb interaction
# V(q) = W₀ * 1/|ql| * tanh(|qD|) * F(q) * F(-q)
# F depends on the Landau level index
# use LLHFSysPara for system parameters
function VFF(q1::Float64, q2::Float64, system::LLHFSysPara,LL::Int64 = 0)::Float64
    D_l = system.D / system.l
    ql = sqrt(q1^2 + q2^2 + 2*system.cosθ * q1*q2) * system.Gl  # |q| in magnetic length units
    if ql == 0.0
        return system.W0 * D_l
    end
    FF_square = laguerrel(LL, 0, ql^2/2)^2   # Landau level form factor squared
    return system.W0 / ql * tanh(ql * D_l) * FF_square
end

# Sign function for reciprocal lattice vectors
# This implements the phase structure of the magnetic translation group
# The sign depends on the parity of the reciprocal lattice vector indices
function ita(g1::Int64, g2::Int64)
    if iseven(g1) && iseven(g2)
        return 1
    else
        return -1
    end
end

# Two-body interaction matrix element
# This implements the full Coulomb interaction with proper magnetic translation phases
# The interaction is computed in momentum space with Landau level projection
# Momentum inputs are Tuple(Float64, Float64) representing (k1, k2) in ratio of Gk
function V_int(kf1, kf2, ki1, ki2, cf1::Int64, cf2::Int64, ci1::Int64, ci2::Int64;
    LL::Int64 = 0, system::LLHFSysPara, Nshell = 2)::ComplexF64

    # valley and large momentum conservation: interaction must conserve component indices
    if ci1 != cf1 || ci2 != cf2
        return 0.0 + 0.0im
    end

    # find the large momentum (1, 2) to be added
    largemomentum_i1 = fld1(ci1, 2)
    largemomentum_i2 = fld1(ci2, 2)
    largemomentum_f1 = fld1(cf1, 2)
    largemomentum_f2 = fld1(cf2, 2)

    ki1 = ki1 .+ (largemomentum_i1 / 3, largemomentum_i1 / 3)
    ki2 = ki2 .+ (largemomentum_i2 / 3, largemomentum_i2 / 3)
    kf1 = kf1 .+ (largemomentum_f1 / 3, largemomentum_f1 / 3)
    kf2 = kf2 .+ (largemomentum_f2 / 3, largemomentum_f2 / 3)

    # Calculate momentum transfer (modulo reciprocal lattice)
    q = rem.(ki1 .- kf1, 1, RoundNearest)
    G_shift1 = round.(Int64, ki1 .- kf1 .- q, RoundNearest)
    G_shift2 = round.(Int64, kf2 .- ki2 .- q, RoundNearest)

    V_total = ComplexF64(0.0)

    # Sum over reciprocal lattice vectors for convergence
    for g1 in -Nshell:Nshell, g2 in -Nshell:Nshell
        if system.cosθ == 0.5 && abs(g1+g2) > Nshell
            continue
        end

        # Construct the full momentum transfer including reciprocal lattice
        qq1 = q[1] + g1
        qq2 = q[2] + g2

        # Calculate phase factors from magnetic translation algebra
        # These phases ensure proper commutation relations and gauge invariance
        phase_angle = 0.5ql_cross(ki1[1], ki1[2], kf1[1], kf1[2])
        phase_angle += 0.5ql_cross(ki1[1]+kf1[1], ki1[2]+kf1[2], qq1, qq2)
        phase_angle += 0.5ql_cross(ki2[1], ki2[2], kf2[1], kf2[2])
        phase_angle += 0.5ql_cross(ki2[1]+kf2[1], ki2[2]+kf2[2], -qq1, -qq2)
        phase = cis(phase_angle)
        # Sign factors from reciprocal lattice vectors
        sign = ita(g1+G_shift1[1], g2+G_shift1[2]) * ita(g1+G_shift2[1], g2+G_shift2[2])

        V_total += sign * phase * VFF(qq1, qq2, system, LL)
    end

    return V_total
end

"""
parameters of ED calculation:
Nk×Nk k-mesh where 3|Nk.
LL -> Landau level index.
k_list[:,k] = [k1; k2] -> "small" momentum = k1/Nk * G1 +  k2/Nk * G2,
where k: 1~k_num.
Dirac point "large" momentum: kD1 = 1/3 * (G1+G2), kD2 = 2/3 * (G1+G2).
mean-field Hamiltonian: Hmf[k,s,s',kD] c†_{k,s,kD} c_{k,s',kD}.
"""
function LLED_init(k_list, H_meanfield, system::LLHFSysPara, 
    N::Int64, LL::Int64 = 0; Nshell::Int64=2)::LLEDPara
    if N%3 !=0 || N < 12
        N = min(12,  N÷3*3)
        @warn "Input k-mesh is too small or not in form of 3n×3n, changed to $N×$N."
    end

    k_num = size(k_list, 2)
    @assert size(H_meanfield) == (2, 2, 2, k_num) "Input mean field size incompatible to k_list"

    v(kf1, kf2, ki1, ki2, cf1, cf2, ci1, ci2) =
        V_int(kf1, kf2, ki1, ki2, cf1, cf2, ci1, ci2;
        LL= LL, system = system, Nshell = Nshell
    )

    return EDPara(
        k_list = k_list, 
        Gk = (0, 0), 
        Nc_hopping = 2, # valley/spin mixed by mean
        Nc_conserve = 2, # Large momentum number conserved
        H_onebody = H_meanfield,
        V_int = v,
    );
    
end





# Below are old mbs procession
#=

function one_body_reduced_density_matrix(psi, block_MBSList; cutoff = 1E-7)

    bits = block_MBSList[1].bits
    sorted_state_num_list = getfield.(block_MBSList, :n)

    index_list = findall( x -> x>cutoff, abs2.(psi))
    psi = psi[index_list]
    sorted_state_num_list = sorted_state_num_list[index_list]

    # DM[i,i′] = <c†_i′ c_i> = <psi | c†_i′ c_i | psi>
    onebodyRDM = zeros(ComplexF64, bits, bits)
    Threads.@threads for i in 0:bits-1
        for i′ in 0:bits-1
            if i == i′
                for index in eachindex(sorted_state_num_list)
                    num = sorted_state_num_list[index]
                    isodd(num >>> i) && (onebodyRDM[1+i,1+i] += abs2(psi[index]))
                end
            elseif i < i′
                for index in eachindex(sorted_state_num_list)
                    num = sorted_state_num_list[index]
                    if isodd(num >>> i) && iseven(num >>> i′)
                        num′ = num - (1 << i) + (1 << i′)
                        index′ = my_searchsortedfirst(sorted_state_num_list, num′)
                        if index′ == 0
                            continue
                        end
                        rhoii′ = conj(psi[index′]) * psi[index]
                        sign_flip = sum(ii -> (num >>> ii)%2, i+1:i′-1; init = 0)
                        isodd(sign_flip) && (rhoii′ = -rhoii′)
                        onebodyRDM[1+i, 1+i′] += rhoii′
                        onebodyRDM[1+i′, 1+i] += conj(rhoii′)
                    end
                end
            end
        end
    end
    return onebodyRDM
end
function reduced_density_matrix(psi, block_MBSList, 
    nA::Vector{Int64}, iA::Vector{Int64}; cutoff = 1E-7)

    bits = block_MBSList[1].bits
    index_list = findall( x -> x>cutoff, abs2.(psi))
    psi = psi[index_list]
    num_list = getfield.(block_MBSList[index_list], :n)

    
    Amask = sum(1 .<< iA; init = 0)
    Bmask = 1 << bits - 1 - Amask
    myless_fine(n1, n2) = n1&Bmask < n2&Bmask || n1&Bmask == n2&Bmask && n1<n2
    myless_coarse(n1, n2) = n1&Bmask < n2&Bmask
    perm = sortperm(num_list; lt = myless_fine)
    psi = psi[perm]
    num_list = num_list[perm]
    Bchunks_lastindices = Int64[0]
    let i=0
        while i < length(num_list)
            i = searchsortedlast(num_list, num_list[i+1]; lt = myless_coarse)
            push!(Bchunks_lastindices, i)
        end
    end

    NA = length(nA)
    RDM_threads = zeros(ComplexF64, NA, NA, Threads.nthreads())
    Threads.@threads for nchunk in 1:length(Bchunks_lastindices)-1
        id = Threads.threadid()
        chunkpiece = Bchunks_lastindices[nchunk]+1:Bchunks_lastindices[nchunk+1]
        numB = num_list[chunkpiece[1]] & Bmask
        
        for i in 1:NA 
            num = numB + nA[i]
            index = my_searchsortedfirst(num_list[chunkpiece], num)
            index == 0 && continue

            RDM_threads[i,i,id] += abs2(psi[chunkpiece[index]])

            for i′ in i+1:NA
                num′ = numB + nA[i′]
                index′ = my_searchsortedfirst(num_list[chunkpiece], num′)
                index′ == 0 && continue
                rhoii′ = conj(psi[chunkpiece[index′]]) * psi[chunkpiece[index]]
                RDM_threads[i, i′, id] += rhoii′
                RDM_threads[i′, i, id] += conj(rhoii′)
            end

        end
    end
    RDM = sum(RDM_threads; dims = 3)[:,:,1]
    return RDM
end

function entanglement_entropy(RDM_A::Matrix{ComplexF64}; cutoff = 1E-6)
    return sum(eigvals(RDM_A)) do x
        if abs(x) < cutoff
            return 0.0
        end
        -x * log2(x)
    end
end


=#


end


