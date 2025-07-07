"""
This module gives the interacting edge calculation with a possible potential V(X)
and the densities of the two ends may be fixed.
works only for lattice G1=(G1,0) and G2=(G2*cosθ, G2*sinθ)
"""
#module LLHFEDGE
    

using MKL
using LinearAlgebra
using TensorCast
using QuadGK
using PhysicalUnits
using MoireIVC.LLHF
using MoireIVC.Basics: ql_cross
using MoireIVC.LLHF: LLHFNumPara, LLHFSysPara, Form_factor

begin
    # qy are integer multiple of G2y/N2
    # qx are in multiple of G1x/N1 but can be real numbers
    function VFF(qx, qy; N1, N2, LL, Gl, D_l, y_x, τn, τn′)

        V  = Form_factor(LL, LL, -qx/N1, -qy/N2*y_x, τn , Gl)
        V *= Form_factor(LL, LL,  qx/N1,  qy/N2*y_x, τn′, Gl)

        if qx==0.0 && qy==0
            V *= D_l
        else
            ql = sqrt((qx/N1)^2 + (qy/N2*y_x)^2 ) * Gl
            V *= tanh(ql*D_l) / ql
        end

        return V
    end
    "<X|kx>, k1,k2 are in unit of G"
    function X_kx_matrixelement(k1, X, τ, k2)
        kx = k1 + 0.5k2
        cispi( kx * (τ*k2 + 2X+1) ) / sqrt(N1)
    end
    # Hartree[Z, Z′, X, X′, τZ, τX, py, ky]
    function Hartree!(
        Hartree::Array{ComplexF64,8}, 
        N1::Int64, N2::Int64, LL::Int64, system::LLHFSysPara;
        Gmax=2,
    )
        Hartree .= 0.0

        Gl = system.Gl
        D_l = system.D / system.l
        y_x = system.ratio12 * system.sinθ

        for Idx in CartesianIndices(Hartree)
            Z  = Idx[1] - 1
            Z′ = Idx[2] - 1
            X  = Idx[3] - 1
            X′ = Idx[4] - 1
            τZ = 3-2Idx[5]
            τX = 3-2Idx[6]
            py = Idx[7] - 1
            ky = Idx[8] - 1
            if ( τZ*(Z′-Z) - τX*(X-X′) ) % N1 == 0
                LxShift_ZX = ( τZ*(Z′-Z) - τX*(X-X′) ) ÷ N1

                LxShift_X, qy = divrem(τX*(X-X′), N1, RoundNearest)
                LxShift_X *= -τX
                qy *= N2

                LxShift_Z = -τZ * (τX * LxShift_X + LxShift_ZX)

                function intgrand(qx)
                    v = VFF(qx, qy; N1=N1, N2=N2, LL=LL, Gl=Gl, D_l=D_l, y_x=y_x, τn=τX, τn′=τZ)
                    return v * cispi(2qx*(0.5LxShift_Z-0.5LxShift_X+0.5(Z+Z′-X-X′)/N1)+(τz*py-τx*ky)/N1/N2)
                end

                
                integral = sum(integrand, -Gmax*N1:Gmax*N1)
                integral *= cispi(-(LxShift_X*ky+LxShift_Z*py)*N1/N2)
                integral *= (-im*τX)^((X-X′-LxShift_X)^2)
                integral *= (-im*τX)^((Z-Z′-LxShift_Z)^2)
                Hartree[Idx] += integral

            end
        end
        return Hartree
    end
    # Fock[Z, Z′, X, X′, τ′, τ, py, ky]
    function Fock(
        Fock::Array{ComplexF64,8}, 
        N1::Int64, N2::Int64, LL::Int64, system::LLHFSysPara
    )
        Fock .= 0.0

        Gl = system.Gl
        D_l = system.D / system.l
        y_x = system.ratio12 * system.sinθ

        for Idx in CartesianIndices(Fock)
            Z  = Idx[1] - 1
            Z′ = Idx[2] - 1
            X  = Idx[3] - 1
            X′ = Idx[4] - 1
            τ′ = 3-2Idx[5]
            τ  = 3-2Idx[6]
            py = Idx[7] - 1
            ky = Idx[8] - 1
            if ( τ′*(X′-Z) - τ*(X-Z′) ) % N1 == 0
                LxShift_′_ = ( τ′*(X′-Z) - τ*(X-Z′) ) ÷ N1

                LxShift, qy = divrem(τ*(X-Z′)*N2+ky-py, N1*N2, RoundNearest)
                LxShift *= -τ

                LxShift′ = -τ′ * (τ * LxShift + LxShift_′_)

                function intgrand(qx)
                    v = VFF(qx, qy; N1=N1, N2=N2, LL=LL, Gl=Gl, D_l=D_l, y_x=y_x, τn=τ, τn′=τ′)
                    return v * cispi(2qx*(0.5LxShift′-0.5LxShift+0.5(Z+X′-X-Z′)/N1)+(τ′-τ)*0.5(ky+py)/N1/N2)
                end

                integral = sum(integrand, (-Gmax*N1:Gmax*N1) .+ 0.5(ky-py)*N1/N2)
                integral *= cispi(-(LxShift*py+LxShift′*ky)*N1/N2)
                integral *= (-im*τ )^((X-Z′-LxShift )^2)
                integral *= (-im*τ′)^((Z-X′-LxShift′)^2)
                Fock[Idx] += integral

            end
        end
        return Fock
    end
    "convert the bulk H0 to edge guiding center representation"
    function H0_buld2edge(num_para::LLHFNumPara)
        # bulk H[k1, k2, τn′, τn] * c†_{k,τn′}c_{k,τn} 
        # edge H[X′, τ′, X, τ, ky] * c†_{k,X′,τ′}c_{k,X,τ} 
        H0_bulk = num_para.H0
        N1 = num_para.N1
        N2 = num_para.N2
        H0_edge = zeros(ComplexF64, N1, 2, N1, 2, N2)
        for ky in 1:N2
            # <X|kx>
            trans = [ X_kx_matrixelement(k1, X, τ, ky)
                for k1 in 0:N1-1, X in 0:N1-1, τ in (1,-1)
            ]

            @reduce sum(X′,τ′,X,τ) H0_edge[X′,τ′,X,τ,$ky] = 
            trans[k1,X′,τ′] * H0_bulk[k1,$ky,τn′,τn] * conj(trans[k1,X,τ])

        end
    end
    @kwdef mutable struct LLEGHFNumPara

        system::LLHFSysPara

        LL::Int64 = 0    # Landau level index
        N1::Int64 = 1    # number of x
        N2::Int64 = 1    # number of ky 
        k_num::Int64 = N1*N2

        # H[X′, τ′, X, τ, ky]
        H0::Array{ComplexF64,5} = zeros(ComplexF64, N1, 2, N1, 2, N2)

        # Hartree[Z, Z′, X, X′, τZ, τX, py, ky]
        BareHartree::Array{ComplexF64,8} = Hartree!(
            zeros(ComplexF64, N1, N1, N1, N1, 2, 2, N2, N2), 
            N1, N2, LL, system,
        )
        # Fock[Z, Z′, X, X′, τ′, τ, py, ky]
        BareFock::Array{ComplexF64,8} = Fock!(
            zeros(ComplexF64, N1, N1, N1, N1, 2, 2, N2, N2), 
            N1, N2, LL, system,
        )
        Hartree::Array{ComplexF64,8} = BareHartree
        Fock::Array{ComplexF64,8} = BareFock

        # α : scale the intervalley interaction 
        # λ : scale down diagonal interaction(HA-HE-XA) 
        α::Float64 = 1.0
        λ::Float64 = 1.0

        # ρ[X, τ, X′, τ′, ky] = <c†_{X′,τn′,ky}c_{X,τn,ky}>
        DMseed::Array{ComplexF64,5} = fill(ComplexF64(0.5), N1, 2, N1, 2, N2)

    end
    "initializa the edge calculation based on bulk calculation"
    function LLEGHF_init(bulkPara::LLHFNumPara)
        para = LLEGHFNumPara(;
            system = bulkPara.system,
            LL = bulkPara.LL,
            N1 = bulkPara.N1,
            N2 = bulkPara.N2,
            H0 = H0_buld2edge(bulkPara)
        )
        if bulkPara.α == NaN
            LLEGHF_change_lambda!(para, bulkPara.λ)
        elseif bulkPara.λ == NaN
            LLEGHF_change_alpha!(para, bulkPara.α)
        else
            error()
        end
    end
    "change α"
    function LLEGHF_change_alpha!(num_para, alpha::Real)
        num_para.λ = NaN
        num_para.α = alpha
        if alpha != 1.0
            num_para.Hartree = copy(num_para.BareHartree)
            num_para.Hartree[:,:,:,:,1,2,:,:] .*= alpha
            num_para.Hartree[:,:,:,:,2,1,:,:] .*= alpha
            num_para.Fock = copy(num_para.BareFock)
            num_para.Fock[:,:,:,:,1,2,:,:] .*= alpha
            num_para.Fock[:,:,:,:,2,1,:,:] .*= alpha
        else
            num_para.Hartree = num_para.BareHartree
            num_para.Fock = num_para.BareFock
        end
        return num_para
    end
    "change λ"
    function LLEGHF_change_lambda!(num_para, lambda::Real)
        num_para.α = NaN
        num_para.λ = lambda
        if lambda != 1.0
            num_para.Hartree = lambda * num_para.BareHartree
            num_para.Fock = copy(num_para.BareFock)
            num_para.Fock[:,:,:,:,1,1,:,:] .*= lambda
            num_para.Fock[:,:,:,:,2,2,:,:] .*= lambda
        else
            num_para.Hartree = num_para.BareHartree
            num_para.Fock = num_para.BareFock
        end
        return num_para
    end
end



"Hartree-Fock interacting mean field generated by density matrix"
function hf_interaction(ρ, para::LLEGHFNumPara)
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




#end



