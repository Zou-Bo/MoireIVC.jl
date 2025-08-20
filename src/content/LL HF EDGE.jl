"""
This module gives the interacting edge calculation with a possible potential V(X)
and the densities of the two ends may be fixed.
works only for lattice G1=(G1,0) and G2=(G2*cosθ, G2*sinθ)
"""
#module LLHFEDGE
    

using MKL
using LinearAlgebra
using TensorCast, TensorOperations
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
    "<X|kx>, k1,k2 are in unit of G, X is in unit of √3/2*a"
    function X_kx_matrixelement(k1, X, τ, k2)
        kx = k1 + 0.5k2
        cispi( kx * (τ*k2/sqrt(0.75) + 2X+1) ) / sqrt(N1)
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

        @time Threads.@threads for Idx in CartesianIndices(Hartree)
            Z  = Idx[1] - 1
            Z′ = Idx[2] - 1
            X  = Idx[3] - 1
            X′ = Idx[4] - 1
            τZ = 3-2Idx[5]
            τX = 3-2Idx[6]
            py = Idx[7] - 1
            ky = Idx[8] - 1
            if ( τZ*(Z′-Z) + τX*(X′-X) ) % N1 == 0
                LxShift0 = ( τZ*(Z′-Z) + τX*(X′-X) ) ÷ N1

                LxShiftZ_1, qy_1 = divrem(τZ*(Z′-Z)*N2, N1*N2, RoundDown)

                for x in -1:0
                    LxShiftZ = LxShiftZ_1 - x
                    qy = qy_1 + x * N1*N2

                    LxShiftZ *= -τZ
                    LxShiftX = -τX * (τZ * LxShiftZ + LxShift0)

                    function integrand(qx)
                        if qy==0 && qx==0.0
                            return 0.0
                        end
                        v = VFF(qx, qy; N1=N1, N2=N2, LL=LL, Gl=Gl, D_l=D_l, y_x=y_x, τn=τX, τn′=τZ)
                        return v * cispi(qx*(LxShiftZ-LxShiftX+(Z+Z′-X-X′)/N1+(τX*ky+τZ*py)/N1/N2))
                    end

                    qx_offset = 0.5isodd(N1*(LxShiftX+X′-X))
                    integral = sum(integrand, (-Gmax*N1:Gmax*N1) .+ qx_offset)

                    integral *= cispi(-(LxShiftX*ky+LxShiftZ*py)*N1/N2)
                    integral *= (1.0im*τX)^((N1*LxShiftX + X′-X)*(N1*LxShiftX + X′+X-1))
                    integral *= (1.0im*τZ)^((N1*LxShiftZ + Z′-Z)*(N1*LxShiftZ + Z′+Z-1))
                    
                    Hartree[Idx] += integral
                end
            end
        end
        return Hartree
    end
    # Fock[Z, Z′, X, X′, τ′, τ, py, ky]
    function Fock!(
        Fock::Array{ComplexF64,8}, 
        N1::Int64, N2::Int64, LL::Int64, system::LLHFSysPara;
        Gmax=2,
    )
        Fock .= 0.0

        Gl = system.Gl
        D_l = system.D / system.l
        y_x = system.ratio12 * system.sinθ

        @time Threads.@threads for Idx in CartesianIndices(Fock)
            Z  = Idx[1] - 1
            Z′ = Idx[2] - 1
            X  = Idx[3] - 1
            X′ = Idx[4] - 1
            τ′ = 3 - 2Idx[5]
            τ  = 3 - 2Idx[6]
            py = Idx[7] - 1
            ky = Idx[8] - 1
            if ( τ′*(X′-Z) + τ*(Z′-X) ) % N1 == 0
                LxShift0 = ( τ′*(X′-Z) + τ*(Z′-X) ) ÷ N1

                LxShift′_1, qy_1 = divrem(τ′*(X′-Z)*N2+ky-py, N1*N2, RoundDown)

                for x in -1:0
                    LxShift′ = LxShift′_1 - x
                    qy = qy_1 + x * N1*N2

                    LxShift′ *= -τ′
                    LxShift = -τ * (τ′ * LxShift′ + LxShift0)

                    function integrand(qx)
                        v = VFF(qx, qy; N1=N1, N2=N2, LL=LL, Gl=Gl, D_l=D_l, y_x=y_x, τn=τ, τn′=τ′)
                        return v * cispi(qx*(LxShift′-LxShift+(Z+X′-X-Z′)/N1+(τ′-τ)*(ky+py)/N1/N2))
                    end

                    qx_offset = rem(0.5(ky-py)*N1/N2+0.5isodd(N1*(LxShift+Z′-X)), 1, RoundNearest)
                    integral = sum(integrand, (-Gmax*N1:Gmax*N1) .+ qx_offset)

                    integral *= cispi(-(LxShift*py+LxShift′*ky)*N1/N2)
                    integral *= (1.0im*τ )^((N1*LxShift + Z′-X)*(N1*LxShift + Z′+X-1))
                    integral *= (1.0im*τ′)^((N1*LxShift′+ X′-Z)*(N1*LxShift′+ X′+Z-1))
                    
                    Fock[Idx] += integral
                end
            end
        end
        return Fock
    end
    "convert the bulk H0 to edge guiding center representation"
    function H0_bulk2edge(num_para::LLHFNumPara)
        # bulk H[k1, k2, τn′, τn] * c†_{k,τn′}c_{k,τn} 
        # edge H[X′, τ′, X, τ, ky] * c†_{k,X′,τ′}c_{k,X,τ} 
        H0_bulk = num_para.H0
        N1 = num_para.N1
        N2 = num_para.N2
        H0_edge = zeros(ComplexF64, N1, 2, N1, 2, N2)
        
        # <X|kx>
        trans = [ X_kx_matrixelement(k1/N1, X, τ, ky/N2)
            for k1 in 0:N1-1, X in 0:N1-1, τ in (1,-1), ky in 0:N2-1
        ]

        @reduce H0_edge[X′,τ′,X,τ,ky] = 
        sum(k1) trans[k1,X′,τ′,ky] * H0_bulk[k1,ky,τ′,τ] * conj(trans[k1,X,τ,ky])

    end
    "convert the bulk density matrix to edge guiding center representation"
    function DM_bulk2edge(DM_bulk, num_para::LLHFNumPara)
        # bulk ρ[k1, k2, τ, τ′] = <c†_{k,τn′}c_{k,τn}>
        # edge ρ[X, τ, X′, τ′, ky] = <c†_{X′,τn′,ky}c_{X,τn,ky}>

        N1 = num_para.N1
        N2 = num_para.N2
        DM_edge = zeros(ComplexF64, N1, 2, N1, 2, N2)
        
        # <X|kx>
        trans = [ X_kx_matrixelement(k1/N1, X, τ, ky/N2)
            for k1 in 0:N1-1, X in 0:N1-1, τ in (1,-1), ky in 0:N2-1
        ]

        @reduce DM_edge[X,τ,X′,τ′,ky] = 
        sum(k1) trans[k1,X′,τ′,ky] * DM_bulk[k1,ky,τ,τ′] * conj(trans[k1,X,τ,ky])

    end
    @kwdef mutable struct LLEGHFNumPara

        system::LLHFSysPara

        LL::Int64 = 0    # Landau level index
        N1::Int64 = 1    # number of x
        N2::Int64 = 1    # number of ky 
        k_num::Int64 = N1*N2

        # H[X′, τ′, X, τ, ky] * c†_{k,X′,τ′}c_{k,X,τ} 
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
    """
    initializa the edge calculation based on bulk calculation.
    optional input: bulk density matrix
    """
    function LLEGHF_init(bulkPara::LLHFNumPara)
        para = LLEGHFNumPara(;
            system = bulkPara.system,
            LL = bulkPara.LL,
            N1 = bulkPara.N1,
            N2 = bulkPara.N2,
            H0 = H0_bulk2edge(bulkPara)
        )
        if isnan(bulkPara.α)
            LLEGHF_change_lambda!(para, bulkPara.λ)
        elseif isnan(bulkPara.λ)
            LLEGHF_change_alpha!(para, bulkPara.α)
        else
            error()
        end
        return para
    end
    function LLEGHF_init(bulkPara::LLHFNumPara, bulkDM)
        para = LLEGHFNumPara(;
            system = bulkPara.system,
            LL = bulkPara.LL,
            N1 = bulkPara.N1,
            N2 = bulkPara.N2,
            H0 = H0_bulk2edge(bulkPara),
            DMseed = DM_bulk2edge(bulkDM, bulkPara)
        )
        if isnan(bulkPara.α)
            LLEGHF_change_lambda!(para, bulkPara.λ)
        elseif isnan(bulkPara.λ)
            LLEGHF_change_alpha!(para, bulkPara.α)
        else
            error()
        end
        return para
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
    # H[X′, τ′, X, τ, ky] * c†_{k,X′,τ′}c_{k,X,τ} 
    # Hartree[Z, Z′, X, X′, τZ, τX, py, ky]
    # Fock[Z, Z′, X, X′, τ′, τ, py, ky]
    # ρ[X, τ, X′, τ′, ky] = <c†_{X′,τn′,ky}c_{X,τn,ky}>
    for τk in 1:2, τp in 1:2
        @tensor H[:, τk, :, τk, :][X′, X, ky] += 
        ρ[:, τp, :, τp, :][Z′, Z, py] * para.Hartree[:,:,:,:,τp,τk,:,:][Z, Z′, X, X′, py, ky]
    end
    for τn in 1:2, τn′ in 1:2
        @tensor H[:, τn′, :, τn, :][X′, X, ky] -= 
        ρ[:, τn′, :, τn, :][Z′, Z, py] * para.Fock[:,:,:,:,τn′, τn,:,:][Z, Z′, X, X′, py, ky]
    end
    return para.system.W0 / para.k_num * H
end
"Tr[ρ*O] expectation value: ρ[X, τ, X′, τ′, ky] * O[X′, τ′, X, τ, ky]"
function trace(rho, O)
    return @reduce sum(X,X′,τ,τ′,ky) rho[X,τ,X′,τ′,ky] * O[X′,τ′,X,τ,ky]
end
"Energy Per Area"
function LLEGHF_EnergyPerArea(ρ; para::LLEGHFNumPara,
    Hint::Array{ComplexF64,5} = hf_interaction(ρ, para), 
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





# only T=0
function hf_onestep!(new_rho, ρ; para::LLEGHFNumPara,
    Hint = hf_interaction(ρ, para), 
    post_procession=[],
    ) 

    new_rho .= 0.0


    # H[X′, τ′, X, τ, ky] * c†_{k,X′,τ′}c_{k,X,τ} 
    # ρ[X, τ, X′, τ′, ky] = <c†_{X′,τn′,ky}c_{X,τn,ky}>
    H = Hint + para.H0

    N1 = size(H,1)

    for ky in axes(H,5)
        vals, vecs = eigen(Hermitian(reshape(H[:,:,:,:,ky], 2N1, 2N1)) )
        for i in 1:N1
            new_rho[:,:,:,:, ky] .+= reshape(vecs[:,i] * vecs[:,i]', N1, 2, N1, 2)
        end
    end

    for f in post_procession
        f(new_rho)
    end

    return
end
function band(ρ; para::LLEGHFNumPara, Hint = hf_interaction(ρ,para), )
    H = Hint + para.H0
    N1 = para.N1; N2 = para.N2
    band = zeros(Float64, 2N1, N2)
    for ky in axes(H,5)
        vals, vecs = eigen(Hermitian(reshape(H[:,:,:,:,ky], 2N1, 2N1)) )
        band[:, ky] .= vals
    end
    return band
end
function hf_converge!(ρ;
    para::LLEGHFNumPara, EPA = LLEGHF_EnergyPerArea,
    error_tolerance = 1E-8, max_iter_times = 200,
    complusive_mixing = false, complusive_mixing_rate = 0.5,
    stepwise_output::Bool = false, final_output::Bool = true,
    post_process_times = max_iter_times, post_procession = [], )


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
function LLEGHF_solve(para, ρ = copy(para.DMseed);
    coherence = 0.0,
    final_procession = [],
    other...
    )

    # we add intervalley coherence into DM
    ρ[:,1,:,2,:] .+= coherence
    ρ[:,2,:,1,:] .+= conj(coherence)
    
    ρ = hf_converge!(ρ; para = para, other...)

    for f in final_procession
        f(ρ) 
    end

    return ρ
end



#end



