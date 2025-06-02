"""
Hartree-Fock on two effective Landau levels with opposite magnetic fields
Two LLs are from two valleys with quasibloch repres in hexagonal BZ
"""
module LLHF

public LLHFNumPara, LLHFSysPara
export LLHF_init_with_alpha, LLHF_init_with_lambda
export LLHF_change_alpha!, LLHF_change_lambda!
export LLHF_EnergyPerArea, LLHF_solve
public Trans, Rot3, PT
public VP_solution, H0_C3_T!, H0_P!, add_phi!, band
public polar_azimuthal_angles, berry_curvature, realspace_pauli



using MKL, LinearAlgebra
using TensorOperations, TensorCast
using PhysicalUnits
using MoireIVC.Basics: LandauLevel_Form_factor, ql_cross, wσ, _γ2


# define the system
begin
    "wavefunction only for τ == 1; do complex conjugate for the other valley"
    function wavefunction0(r1, r2, k1, k2, n::Int64=0; a1, a2, l, γ2, norm)
        a1c = (a1[1] + im*a1[2]) / l
        a2c = (a2[1] + im*a2[2]) / l
        tau = a2c/a1c
        r = r1 + r2*tau
        k = k2 - k1*tau # effective real-space shift by momentum

        σ = a1c * wσ(r-k, tau) * exp( - 0.5γ2 * (r-k)^2 )
        e1 = exp(conj(k) * r * abs2(a1c) / 2.0)
        e2 = exp( - (abs2(r)+abs2(k)) * abs2(a1c) / 4.0)
        return σ*e1*e2/norm
    end
    function WF_normalizer(a1, a2, l, γ2)
        Nr = 100
        wf = [ abs2(wavefunction0(r1/Nr, r2/Nr, 0., 0., 0;
                a1 = a1, a2 = a2, l = l, γ2 = γ2, norm = 1.0)
            ) 
            for r1 in 0:Nr-1, r2 in 0:Nr-1
        ]
        int = sum(wf) / Nr^2 * 2π*l^2
        return sqrt(int)
    end
    @kwdef mutable struct LLHFSysPara
        # material constants
        e::Float64 = ElementaryCharge
        m_e::Float64 = ElectronMass * 0.4
        ϵ::Float64
        # geometry parameters
        a_Moire::Float64
        ratio12::Float64 = 1.0
        cosθ::Float64 = 0.5  # angle between G1 and G2
        sinθ::Float64 = sqrt(0.75)
        Area_uc::Float64 = sinθ * a_Moire^2 /ratio12
        D::Float64             # gate distance for screening
        # effective B field
        l::Float64 = sqrt(Area_uc / (2π))
        B::Float64 = 1.0 / (e * l^2)
        ω_c::Float64 = e*B/m_e
        W0::Float64 = e^2/ϵ/l
        # real and reciprocal lattice vectors
        a1::Vector{Float64} = [ sinθ; -cosθ] * a_Moire
        a2::Vector{Float64} = [   0.;    1.] * a_Moire / ratio12
        G_Moire::Float64 = 2π / a_Moire / sinθ
        G1::Vector{Float64} = [  1.;   0.] * G_Moire
        G2::Vector{Float64} = [cosθ; sinθ] * G_Moire * ratio12
        Gl::Float64 = G_Moire * l 
        # wavefunction normalization
        γ2::ComplexF64 = _γ2(a1, a2)
        WF_nmlz::Float64 = WF_normalizer(a1, a2, l, γ2)
    end
    "twist angle θ in degree, gate screening, triangular lattice"
    function define_MoTe2system(twist_θ::Float64 = 2.1, D::Float64 = 20nm;
        ϵ = 5.0, a_mono = 0.352nm)
        a_Moire = a_mono / 2.0 / sind(twist_θ / 2.0)
        return LLHFSysPara(ϵ = ϵ, a_Moire = a_Moire, D = D, )
    end
    function wavefunction(r1, r2, k1, k2, n::Int64=0; sys_para::LLHFSysPara)

        return wavefunction0(r1, r2, k1, k2, n; 
            a1=sys_para.a1, a2=sys_para.a2, l=sys_para.l, 
            γ2=sys_para.γ2, norm=sys_para.WF_nmlz
        )
    end
    "Landau level form factor, τ=±1"
    function Form_factor(n_left::Int64, n_right::Int64, qx::Float64, qy::Float64, 
        τ::Int64, l::Float64)

        return LandauLevel_Form_factor(
            n_left, n_right, qx, qy; τ=τ, l=l
        )
    end
end

# define all the nemerical parameters
# H[k1, k2, τn′, τn] * c†_{k,τn′}c_{k,τn}
# ρ[k1, k2, τ, τ′] = <c†_{k,τn′}c_{k,τn}>
begin
    # interaction V(q) / (2pi l²) =  W0 * following function
    function V_int(qq1, qq2; N1, N2, Gl, r12, D_l, cosθ)
        if qq1==0 && qq2==0
            V = D_l
        else
            ql = sqrt((qq1/N1)^2 + (qq2/N2*r12)^2 + 2cosθ*(qq1/N1)*(qq2/N2*r12)) * Gl
            V = 1.0 / ql * tanh(ql*D_l)
        end
        return V
    end
    # Hartree[p1, p2, k1, k2, τp, τk]
    function Hartree!(Hartree,N1, N2, LL, sys_para::LLHFSysPara; Nshell=2)
        Hartree .= 0.0

        # N shells of reciprocal lattice vectors G
        for g1 in -Nshell:Nshell, g2 in -Nshell:Nshell
            if g1==0 && g2==0
                continue
            elseif abs(g1+g2)>Nshell
                continue
            end
            V = V_int(g1*N1, g2*N2; N1=N1, N2=N2, r12 = sys_para.ratio12,
                Gl=sys_para.Gl, D_l=sys_para.D/sys_para.l, cosθ=sys_para.cosθ
            )

            for τp = [1;-1], τk = [1;-1]
                phase = [cis(ql_cross((τk*k1 - τp*p1)/N1, (τk*k2 - τp*p2)/N2, g1, g2) )
                    for p1 in 0:N1-1, p2 in 0:N2-1, k1 in 0:N1-1, k2 in 0:N2-1
                ]
                VFF = Form_factor(LL,LL, (-g1*sys_para.G1-g2*sys_para.G2)..., τk, sys_para.l) * 
                    Form_factor(LL,LL, (g1*sys_para.G1+g2*sys_para.G2)..., τp, sys_para.l) * V
                Hartree[:,:,:,:, (3-τp)÷2, (3-τk)÷2] .+= VFF .* phase 
            end
        end
        return Hartree
    end
    # Fock[p1, p2, k1, k2, τn′, τn]
    function Fock!(Fock, N1, N2, LL, sys_para::LLHFSysPara; Nshell=2)
        Fock .= 0.0

        Threads.@threads for p1 in 0:N1-1    
        for (p2,k1,k2) in Iterators.product(0:N2-1, 0:N1-1, 0:N2-1)
            q1 = k1 - p1
            q2 = k2 - p2
            q1 %= N1
            q2 %= N2
            q1 < -N1 ÷ 2 && (q1 += N1)
            q1 >  N1 ÷ 2 && (q1 -= N1)
            q2 < -N2 ÷ 2 && (q2 += N2)
            q2 >  N2 ÷ 2 && (q2 -= N2)
            # N shells of reciprocal lattice vectors G
            for g1 in -Nshell:Nshell, g2 in -Nshell:Nshell
                if abs(g1+g2)>Nshell
                    continue
                end

                qq1 = q1 + g1 * N1
                qq2 = q2 + g2 * N2

                V = V_int(qq1, qq2; N1=N1, N2=N2, r12 = sys_para.ratio12,
                    Gl=sys_para.Gl, D_l=sys_para.D/sys_para.l, cosθ=sys_para.cosθ
                )

                phase_angle = ql_cross(k1/N1, k2/N2, p1/N1, p2/N2)
                phase_angle += ql_cross((k1+p1)/N1, (k2+p2)/N2, qq1/N1, qq2/N2)

                for τn′ = [1;-1], τn = [1;-1]

                    factor = τn == τn′ ? 1.0 : cis(τn * phase_angle)

                    VFF = Form_factor(LL,LL, (-qq1*sys_para.G1/N1-qq2*sys_para.G2/N2)..., τn, sys_para.l) * 
                        Form_factor(LL,LL, (qq1*sys_para.G1/N1+qq2*sys_para.G2/N2)..., τn′, sys_para.l) * V
                    Fock[1+p1,1+p2,1+k1,1+k2, (3-τn′)÷2, (3-τn)÷2] += factor * VFF
                end
            end
        end
        end
        return Fock
    end
    @kwdef mutable struct LLHFNumPara

        system::LLHFSysPara
        
        LL::Int64 = 0    # Landau level index
        N1::Int64 = 1
        N2::Int64 = 1
        k_num::Int64 = N1*N2

        # H[k1, k2, τn′, τn] * c†_{k,τn′}c_{k,τn} 
        H0::Array{ComplexF64,4} = zeros(ComplexF64, N1, N2, 2, 2)

        # Hartree[p1, p2, k1, k2, τp, τk]
        BareHartree::Array{ComplexF64,6} = Hartree!(
            zeros(ComplexF64, N1, N2, N1, N2, 2, 2), 
            N1, N2, LL, system,
        )
        # Fock[p1, p2, k1, k2, τn′, τn]
        BareFock::Array{ComplexF64,6} = Fock!(
            zeros(ComplexF64, N1, N2, N1, N2, 2, 2), 
            N1, N2, LL, system,
        )
        Hartree::Array{ComplexF64,6} = BareHartree
        Fock::Array{ComplexF64,6} = BareFock

        # α : scale the intervalley interaction 
        # λ : scale down diagonal interaction(HA-HE-XA) 
        α::Float64 = 1.0
        λ::Float64 = 1.0

        # ρ[k1, k2, τ, τ′] = <c†_{k,τn′}c_{k,τn}>
        DMseed::Array{ComplexF64,4} = fill(ComplexF64(0.5), N1, N2, 2, 2)

    end
    "initializa the numerical calculation using α"
    function LLHF_init_with_alpha(alpha::Real, sys_para::LLHFSysPara = define_MoTe2system(); others...)
        num_para = LLHFNumPara(; system = sys_para, others...)
        return LLHF_change_alpha!(num_para, alpha)
    end
    "initializa the numerical calculation using λ"
    function LLHF_init_with_lambda(lambda::Real, sys_para::LLHFSysPara = define_MoTe2system(); others...)
        num_para = LLHFNumPara(; system = sys_para, others...)
        return LLHF_change_lambda!(num_para, lambda)
    end
    "change α"
    function LLHF_change_alpha!(num_para, alpha::Real)
        num_para.λ = NaN
        num_para.α = alpha
        if alpha != 1.0
            num_para.Hartree = copy(num_para.BareHartree)
            num_para.Hartree[:,:,:,:,1,2] .*= alpha
            num_para.Hartree[:,:,:,:,2,1] .*= alpha
            num_para.Fock = copy(num_para.BareFock)
            num_para.Fock[:,:,:,:,1,2] .*= alpha
            num_para.Fock[:,:,:,:,2,1] .*= alpha
        else
            num_para.Hartree = num_para.BareHartree
            num_para.Fock = num_para.BareFock
        end
        return num_para
    end
    "change λ"
    function LLHF_change_lambda!(num_para, lambda::Real)
        num_para.α = NaN
        num_para.λ = lambda
        if lambda != 1.0
            num_para.Hartree = lambda * num_para.BareHartree
            num_para.Fock = copy(num_para.BareFock)
            num_para.Fock[:,:,:,:,1,1] .*= lambda
            num_para.Fock[:,:,:,:,2,2] .*= lambda
        else
            num_para.Hartree = num_para.BareHartree
            num_para.Fock = num_para.BareFock
        end
        return num_para
    end
    "τ = 1 or 2"
    function VP_solution(num_para::LLHFNumPara, τ)
        DM = zeros(ComplexF64, size(num_para.DMseed))
        DM[:,:,τ,τ] .= 1.0
        return DM
    end
    "sinoal H0 with C3 symmetry that opens a gap in unit of W0"
    function H0_C3_T!(num_para, gap=0.1)
        num_para.H0 .= gap * [num_para.system.W0 * sqrt(4/27) *
            (sin(-2π*k2/num_para.N2)+sin(-2π*k1/num_para.N1)+sin(2π*(k1/num_para.N1+k2/num_para.N2)))*
            (1.5-τn)*(τn′==τn) 
                for k1 in 0:num_para.N1-1, k2 in 0:num_para.N2-1, τn′ in 1:2, τn in 1:2
        ];
        return num_para
    end
    "constant H0 that opens a gap in unit of W0"
    function H0_P!(num_para, gap=0.1)
        num_para.H0 .= gap * [num_para.system.W0*(1.5-τn)*(τn′==τn) 
            for k1 in 0:num_para.N1-1, k2 in 0:num_para.N2-1, τn′ in 1:2, τn in 1:2
        ];
        return num_para
    end
end


"Hartree-Fock interacting mean field generated by density matrix"
function hf_interaction(ρ, para::LLHFNumPara)
    H = zeros(ComplexF64, size(para.H0))
    for τk in 1:2, τp in 1:2
        @tensor H[:,:, τk, τk][k1, k2] += 
        ρ[:,:, τp, τp][p1, p2] * para.Hartree[:,:,:,:,τp,τk][p1, p2, k1, k2]
    end
    for τn in 1:2, τn′ in 1:2
        @tensor H[:,:, τn′, τn][k1, k2] -= 
        ρ[:,:, τn′, τn][p1, p2] * para.Fock[:,:,:,:,τn′,τn][p1, p2, k1, k2]
    end
    return para.system.W0 / para.k_num * H
end
function hf_interaction(ρ, para::LLHFNumPara, part::Symbol)
    H = zeros(ComplexF64, size(para.H0))
    if part == :HA
        for τk in 1:2
            τp = τk
            @tensor H[:,:, τk, τk][k1, k2] += 
            ρ[:,:, τp, τp][p1, p2] * para.Hartree[:,:,:,:,τp,τk][p1, p2, k1, k2]
        end
    elseif part == :HE
        for τk in 1:2
            τp = 3-τk
            @tensor H[:,:, τk, τk][k1, k2] += 
            ρ[:,:, τp, τp][p1, p2] * para.Hartree[:,:,:,:,τp,τk][p1, p2, k1, k2]
        end
    elseif part == :XA
        for τn in 1:2
            τn′ = τn
            @tensor H[:,:, τn′, τn][k1, k2] -= 
            ρ[:,:, τn′, τn][p1, p2] * para.Fock[:,:,:,:,τn′,τn][p1, p2, k1, k2]
        end
    elseif part == :XE
        for τn in 1:2
            τn′ = 3-τn
            @tensor H[:,:, τn′, τn][k1, k2] -= 
            ρ[:,:, τn′, τn][p1, p2] * para.Fock[:,:,:,:,τn′,τn][p1, p2, k1, k2]
        end
    else
        error("part can onl be :HA, :HE, :XA, :XE")
    end
    return para.system.W0 / para.k_num * H
end
"Tr[ρ*O] expectation value: ρ[k1, k2, τ, τ′] * O[k1, k2, τ′, τ]"
function trace(rho, O)
    #return @tensor rho[k1, k2, τ, τ′] * O[k1, k2, τ′, τ]
    return @reduce sum(k1,k2,τ,τ′) rho[k1,k2,τ,τ′] * O[k1,k2,τ′,τ]
end
"Energy Per Area"
function LLHF_EnergyPerArea(ρ; para::LLHFNumPara,
    Hint::Array{ComplexF64,4} = hf_interaction(ρ, para), 
    imag_print = false, warn = true, )

    # int_energy = trace(ρ, Hint) / 2.0
    # kin_energy = trace(ρ, H0)
    # EPA = (kin_energy + int_energy) / k_num / (2π*l^2)
    EPA = trace(ρ, para.H0 + 0.5Hint) / para.k_num / para.system.Area_uc
    if imag_print
        println("EPA is ", EPA)
    end
    if warn && abs(imag(EPA)) > 1e-5
        @warn "Energy per area $EPA has large imaginary part."
    end
    return  real(EPA)
end






# post-processes that enforce the symmetry
module CrystalSym

    using MKL, LinearAlgebra
    using MoireIVC.Basics: ql_cross
    export Trans, Rot3, PT




    "enforce translation symmetry with G/2"
    struct Trans 
        g1::Int64
        g2::Int64
    end
    function (Ts::Trans)(rho)

        N1 = size(rho, 1)
        N2 = size(rho, 2)
        g1 = mod(Ts.g1, 4)
        g2 = mod(Ts.g2, 4)
        if isodd(g1*N1) || isodd(g2*N2)
            @warn "translation symmetry cannot be forced due to odd (N1=$N1, N2=$N2)" 
            return
        end
        if iseven(g1) && iseven(g2)
            @warn "the given translation symmetry ($g1, $g2) is empty"
            return
        end
        boost1, boost2 = (g1*N1÷2, g2*N2÷2)

        identification = falses(N1,N2) # true if this element is used
        for k1 in 0:N1-1, k2 in 0:N2-1
            if identification[1+k1, 1+k2] 
                continue
            else
                identification[1+k1, 1+k2] = true
            end

            # Tk = k + G/2 (mod G)
            Tk1 = k1 + boost1
            Tk2 = k2 + boost2
            TG1, Tk1 = fldmod(Tk1, N1)
            TG2, Tk2 = fldmod(Tk2, N2)
            identification[1+Tk1, 1+Tk2] = true

            phase = ql_cross(boost1/N1, boost2/N2, k1/N1, k2/N2) - ql_cross(TG1, TG2, Tk1/N1, Tk2/N2) + π

            rho[1+k1,1+k2,1,2] = (rho[1+k1,1+k2,1,2] + rho[1+Tk1,1+Tk2,1,2]*cis( phase)) / 2.
            rho[1+k1,1+k2,2,1] = (rho[1+k1,1+k2,2,1] + rho[1+Tk1,1+Tk2,2,1]*cis(-phase)) / 2.
            rho[1+k1,1+k2,1,1] = (rho[1+k1,1+k2,1,1] + rho[1+Tk1,1+Tk2,1,1] ) / 2.
            rho[1+k1,1+k2,2,2] = (rho[1+k1,1+k2,2,2] + rho[1+Tk1,1+Tk2,2,2] ) / 2.

            rho[1+Tk1,1+Tk2,1,2] = rho[1+k1,1+k2,1,2] * cis(-phase)
            rho[1+Tk1,1+Tk2,2,1] = rho[1+k1,1+k2,2,1] * cis(phase)
            rho[1+Tk1,1+Tk2,1,1] = rho[1+k1,1+k2,1,1]
            rho[1+Tk1,1+Tk2,2,2] = rho[1+k1,1+k2,2,2]
        end
    end


    "enforce C3 rotational symmetry with order n around k=0"
    struct Rot3 
        n::Int64
    end
    function (C3::Rot3)(rho)

        N1 = size(rho, 1)
        N2 = size(rho, 2)
        n = mod(C3.n, 3)
        if N1 !=N2
            @warn "Rotation symmetry cannot be forced due to different N1=$N1 and N2=$N2" 
            return
        end

        identification = falses(N1,N2) # true if this element is used
        for k1 in 0:N1-1, k2 in 0:N2-1
            if identification[1+k1, 1+k2] 
                continue
            else
                identification[1+k1, 1+k2] = true
            end

            # Rk = Rot(2π/3) * k 
            # R2k = Rot(2π/3) * Rk
            Rk1, Rk2 = [-1 -1; 1 0] * [k1; k2]
            R2k1, R2k2 = [-1 -1; 1 0] * [Rk1; Rk2]
            RG1, Rk1 = fldmod(Rk1, N1)
            RG2, Rk2 = fldmod(Rk2, N2)
            R2G1, R2k1 = fldmod(R2k1, N1)
            R2G2, R2k2 = fldmod(R2k2, N2)
            identification[1+Rk1, 1+Rk2] = true
            identification[1+R2k1, 1+R2k2] = true

            phaseR = -ql_cross(RG1, RG2, Rk1/N1, Rk2/N2) + n*2π/3
            phaseR2 = -ql_cross(R2G1, R2G2, R2k1/N1, R2k2/N2) - n*2π/3

            rho[1+k1,1+k2,1,2] = (rho[1+k1,1+k2,1,2] + rho[1+Rk1,1+Rk2,1,2]*cis( phaseR) + rho[1+R2k1,1+R2k2,1,2]*cis( phaseR2)) / 3.
            rho[1+k1,1+k2,2,1] = (rho[1+k1,1+k2,2,1] + rho[1+Rk1,1+Rk2,2,1]*cis(-phaseR) + rho[1+R2k1,1+R2k2,2,1]*cis(-phaseR2)) / 3.
            rho[1+k1,1+k2,1,1] = (rho[1+k1,1+k2,1,1] + rho[1+Rk1,1+Rk2,1,1]              + rho[1+R2k1,1+R2k2,1,1]              ) / 3.
            rho[1+k1,1+k2,2,2] = (rho[1+k1,1+k2,2,2] + rho[1+Rk1,1+Rk2,2,2]              + rho[1+R2k1,1+R2k2,2,2]              ) / 3.

            rho[1+Rk1,1+Rk2,1,2] = rho[1+k1,1+k2,1,2] * cis(-phaseR)
            rho[1+Rk1,1+Rk2,2,1] = rho[1+k1,1+k2,2,1] * cis( phaseR)
            rho[1+Rk1,1+Rk2,1,1] = rho[1+k1,1+k2,1,1]
            rho[1+Rk1,1+Rk2,2,2] = rho[1+k1,1+k2,2,2]

            rho[1+R2k1,1+R2k2,1,2] = rho[1+k1,1+k2,1,2] * cis(-phaseR2)
            rho[1+R2k1,1+R2k2,2,1] = rho[1+k1,1+k2,2,1] * cis( phaseR2)
            rho[1+R2k1,1+R2k2,1,1] = rho[1+k1,1+k2,1,1]
            rho[1+R2k1,1+R2k2,2,2] = rho[1+k1,1+k2,2,2]
        end
    end

    
    struct PT 
        n::Int64
        sym::Symbol
    end
    PT(n::Int64) = PT(n, :none)
    function (Inv::PT)(rho)

        N1 = size(rho, 1)
        N2 = size(rho, 2)
        n = mod(Inv.n, 2)

        identification = falses(N1,N2) # true if this element is used
        for k1 in 0:N1-1, k2 in 0:N2-1
            if identification[1+k1, 1+k2] 
                continue
            else
                identification[1+k1, 1+k2] = true
            end

            # Mk = -k (mod G)
            Mk1 = -k1
            Mk2 = -k2
            MG1, Mk1 = fldmod(Mk1, N1)
            MG2, Mk2 = fldmod(Mk2, N2)
            identification[1+Mk1, 1+Mk2] = true

            phase = n*π - ql_cross(MG1, MG2, Mk1/N1, Mk2/N2)

            rho[1+k1,1+k2,1,2] = (rho[1+k1,1+k2,1,2] + rho[1+Mk1,1+Mk2,1,2]*cis( phase) ) / 2.
            rho[1+k1,1+k2,2,1] = (rho[1+k1,1+k2,2,1] + rho[1+Mk1,1+Mk2,2,1]*cis(-phase) ) / 2.

            rho[1+Mk1,1+Mk2,1,2] = rho[1+k1,1+k2,1,2] * cis(-phase)
            rho[1+Mk1,1+Mk2,2,1] = rho[1+k1,1+k2,2,1] * cis( phase)

            if Inv.sym == :T
                rho[1+k1,1+k2,1,1] = (rho[1+k1,1+k2,1,1] + rho[1+Mk1,1+Mk2,2,2] )/2.
                rho[1+k1,1+k2,2,2] = (rho[1+k1,1+k2,2,2] + rho[1+Mk1,1+Mk2,1,1] )/2.
                rho[1+Mk1,1+Mk2,1,1] = rho[1+k1,1+k2,2,2]
                rho[1+Mk1,1+Mk2,2,2] = rho[1+k1,1+k2,1,1]
            elseif Inv.sym == :P
                rho[1+k1,1+k2,1,1] = (rho[1+k1,1+k2,1,1] + rho[1+Mk1,1+Mk2,1,1] )/2.
                rho[1+k1,1+k2,2,2] = (rho[1+k1,1+k2,2,2] + rho[1+Mk1,1+Mk2,2,2] )/2.
                rho[1+Mk1,1+Mk2,1,1] = rho[1+k1,1+k2,1,1]
                rho[1+Mk1,1+Mk2,2,2] = rho[1+k1,1+k2,2,2]
            elseif Inv.sym == :PT
                rho[1+k1,1+k2,1,1] = 0.5
                rho[1+k1,1+k2,2,2] = 0.5
                rho[1+Mk1,1+Mk2,1,1] = 0.5
                rho[1+Mk1,1+Mk2,2,2] = 0.5
            else
                # not to do anything for other markers
            end

        end
    end


end
using .CrystalSym



# only T=0
function hf_onestep!(new_rho, ρ; para::LLHFNumPara,
    Hint = hf_interaction(ρ, para), 
    post_procession=[],
    ) 
    # H[k1, k2, τn′, τn]
    # ρ[k1, k2, τ, τ′]
    H = Hint + para.H0

    for k1 in axes(H,1), k2 in axes(H,2)
        vals, vecs = eigen(Hermitian(H[k1,k2,:,:]) )
        new_rho[k1,k2,:,:] .= vecs[:,1] * vecs[:,1]'
    end

    for f in post_procession
        f(new_rho)
    end

    return
end
function band(ρ; para::LLHFNumPara, Hint = hf_interaction(ρ,para), )
    H = Hint + para.H0
    N1 = para.N1; N2 = para.N2
    band = zeros(Float64, N1, N2, 2)
    for k1 in 0:N1-1, k2 in 0:N2-1
        vals, vecs = eigen(Hermitian(H[1+k1,1+k2,:,:]) )
        band[1+k1, 1+k2, :] .= vals
    end
    return band
end
function hf_converge!(ρ;
    para::LLHFNumPara, EPA = LLHF_EnergyPerArea,
    error_tolerance = 1E-8, max_iter_times = 200,
    complusive_mixing = false, complusive_mixing_rate = 0.5,
    stepwise_output::Bool = false, final_output::Bool = true,
    post_process_times = 0, post_procession = [], )


    if complusive_mixing
        println("Iteration uses complusive mixing rate $complusive_mixing_rate.")
    end
    
    hint = hf_interaction(ρ, para)

    new_rho = similar(ρ)
    new_hint = similar(hint)

    for i in 1:max_iter_times
        if i == post_process_times+1
            empty!(post_procession)
        end
        hf_onestep!(new_rho, ρ; Hint = hint, 
            post_procession = post_procession, para = para
        )
        
        new_hint .= hf_interaction(new_rho, para)

        error = maximum(abs.(new_rho-ρ))

        E0 = EPA(ρ, Hint = hint, para=para, imag_print=false)
        stepwise_output && println("$i \t DM error:$error \t E/S:$E0")
        E1 = EPA(new_rho, Hint = new_hint, para=para, imag_print=false)
        Eh = EPA((ρ+new_rho)/2., para=para, imag_print=false) # h stands for halfway
        k = E0+E1-2Eh

        if complusive_mixing
            if complusive_mixing_rate == 0.5
                stepwise_output && println("update rate = 0.5: unstable solution with higher energy")
                ρ .= (ρ+new_rho)/2.
                hint = hf_interaction(ρ, para)
            else
                x = complusive_mixing_rate
                stepwise_output && println("update rate = $(round(x; digits=2))")
                ρ .+= x*(new_rho - ρ)
                hint = hf_interaction(ρ, para)
            end
        else
            if k <= 0. 
                if E0 >= E1
                stepwise_output && println("update rate = 1.0: unstable solution with lower energy")
                ρ .= new_rho
                hint .= new_hint
                else
                    stepwise_output && println("update rate = 0.5: unstable solution with higher energy")
                    ρ .= (ρ+new_rho)/2.
                    hint = hf_interaction(ρ, para)
                end
            elseif k <= 2(E0-E1)
                stepwise_output && println("update rate = 1.0: energy minimum is ahead but we constrained to 1.0")
                ρ .= new_rho
                hint .= new_hint
            else
                x0 = 0.5 + (E0-E1)/k
                x = max(0.1, x0)
                stepwise_output && println("update rate = $(round(x; digits=2)): energy minimum at $(round(x0; digits=2))")
                ρ .+= x*(new_rho - ρ)
                hint = hf_interaction(ρ, para)
            end
        end


        if error <= error_tolerance
            final_output && println("converged in $i iterations, density error = $error")
            break
        end
        if i == max_iter_times
            final_output && println("not converged after $max_iter_times iterations, density error = $error.")
        end
    end

    return ρ
end
function add_phi!(rho, phi = angle(rho[1,1,1,2]))
    rho[:,:, 1, 2] .*= cis(-phi)
    rho[:,:, 2, 1] .*= cis(+phi)
    return rho
end
function LLHF_solve(para, ρ = copy(para.DMseed);
    coherence = 0.0,
    final_procession = [add_phi!],
    other...
    )

    # we add intervalley coherence into DM
    ρ[:,:,1,2] .+= coherence
    ρ[:,:,2,1] .+= conj(coherence)
    
    ρ = hf_converge!(ρ; para = para, other...)

    for f in final_procession
        f(ρ) 
    end

    return ρ
end



# θ/π and ϕ/π
function polar_azimuthal_angles(rho, para::LLHFNumPara)
    N1 = para.N1
    N2 = para.N2
    theta = Matrix{Float64}(undef, N1,N2);
    phi = Matrix{Float64}(undef, N1,N2);
    for k2 = 1:N2, k1 = 1:N1
        costheta = 2rho[k1,k2,1,1].re-1
        if costheta >= 1.
            theta[k1, k2] =  0.
        elseif costheta <= -1.
            theta[k1, k2] = pi
        else
            theta[k1, k2] = acos(costheta)
        end
        phi[k1, k2] = angle(rho[k1,k2,2,1])
    end
    return theta, phi
end
# Berry Curvature
function berry_curvature(rho, para::LLHFNumPara)

    N1 = para.N1
    N2 = para.N2

    θ = zeros(Float64, N1+1, N2+1)
    ϕ = zeros(Float64, N1+1, N2+1)
    intact_rho = similar(rho)
    hf_onestep!(intact_rho, rho; para = para, Hint = hf_interaction(rho, para))
    local theta, phi = polar_azimuthal_angles(intact_rho, para)
    for k1 in 0:N1, k2 in 0:N2
        g1 = Int64(k1==N1)
        g2 = Int64(k2==N2)
        θ[1+k1, 1+k2] = theta[1+k1-g1*N1, 1+k2-g2*N2]
        ϕ[1+k1, 1+k2] = phi[1+k1-g1*N1, 1+k2-g2*N2]
        ϕ[1+k1, 1+k2] += 2π * (g1 * k2/N2 - g2 * k1/N1) #- π * (g1+g2-g1*g2)
        if ϕ[1+k1, 1+k2] > π
            ϕ[1+k1, 1+k2] -= 2π
        elseif ϕ[1+k1, 1+k2] <= -π
            ϕ[1+k1, 1+k2] += 2π
        end
    end
    ψ1 = cos.(0.5θ) .* cis.(-0.5ϕ)
    ψ2 = sin.(0.5θ) .* cis.(+0.5ϕ)
    
    # phases are in units of 2π
    inn_prod1  =  conj.(ψ1[begin+1:end, :]) .* ψ1[begin:end-1, :] .* cispi.(+collect(0:N2)'/N1/N2) # .* F(δk)
    inn_prod1 .+= conj.(ψ2[begin+1:end, :]) .* ψ2[begin:end-1, :] .* cispi.(-collect(0:N2)'/N1/N2) # .* F(δk)
    phase1 = angle.(inn_prod1) / 2π
    inn_prod2  =  conj.(ψ1[:, begin+1:end]) .* ψ1[:, begin:end-1] .* cispi.(-collect(0:N1)/N1/N2) # .* F(δk)
    inn_prod2 .+= conj.(ψ2[:, begin+1:end]) .* ψ2[:, begin:end-1] .* cispi.(+collect(0:N1)/N1/N2) # .* F(δk)
    phase2 = angle.(inn_prod2) / 2π
    
    Ω = phase1[:,begin:end-1] + phase2[begin+1:end,:] - phase1[:,begin+1:end] - phase2[begin:end-1,:]
    for k1 in 1:N1, k2 in 1:N2
        while Ω[k1, k2] > 0.5
            Ω[k1, k2] -= 1.
        end
        while Ω[k1, k2] <= -0.5
            Ω[k1, k2] += 1.
        end
    end

    return Ω
end
function realspace_pauli(rx, ry, ρ, para::LLHFNumPara)
    r1 = [rx; ry] ⋅ para.system.G1 / 2π
    r2 = [rx; ry] ⋅ para.system.G2 / 2π
    N1 = para.N1; N2 = para.N2
    WF = Array{ComplexF64,3}(undef, N1, N2, 2)
    for k1 in 0:N1-1, k2 in 0:N2-1
        WF[1+k1, 1+k2, 1] = wavefunction(r1, r2, k1/N1, k2/N2, 0; sys_para = para.system)
        WF[1+k1, 1+k2, 2] = conj(wavefunction(r1, r2, -k1/N1, -k2/N2, 0; sys_para = para.system))
    end
    local_density = zeros(ComplexF64, 2, 2)
    for τ in 1:2, τ′ in 1:2
        for k1 in 1:N1, k2 in 1:N2
            local_density[τ,τ′] += ρ[k1,k2,τ,τ′] * WF[k1,k2,τ] * conj(WF[k1,k2,τ′])
        end
    end
    local_density ./= para.k_num
    local_density .*= para.system.Area_uc
    n  = real(local_density[1,1] + local_density[2,2])
    sz = real(local_density[1,1] - local_density[2,2])
    sx = 2real(local_density[1,2])
    sy = -2imag(local_density[1,2])
    return (sx, sy, sz, n)
end



    
end