"""
Do Time-dependent Hartree-Fock on LLHF results.
"""
module LLTDHF

public LLTDHFGroundState, TDHF_V_matrix_spin
export TDHF_groundstateanalysis
export TDHF_ZEXV, TDHF_solve


using MKL, LinearAlgebra
using TensorOperations, KrylovKit
using MoireIVC.LLHF
using MoireIVC.LLHF: LLHFNumPara, Form_factor, V_int
using MoireIVC.Basics: ql_cross
# using PhysicalUnits

@kwdef struct LLTDHFGroundState
    HFpara::LLHFNumPara
    band::Array{Float64, 3}
    eigwf::Array{ComplexF64, 4}
end



"""
representation transform s->n : 
covariant c†_n = <s|n> c†_s = eigwf[s,n] c†_s
contravariant c_n = <n|s> c_s = conj(eigwf[s,n]) c_s
"""
function TDHF_groundstateanalysis(ρ, HFpara::LLHFNumPara)
    H = LLHF.hf_interaction(ρ, HFpara) + HFpara.H0
    N1 = HFpara.N1; N2 = HFpara.N2
    band = zeros(Float64, 2, N1, N2)
    eigwf = zeros(ComplexF64, 2, 2, N1, N2)
    for k1 in axes(H,1), k2 in axes(H,2)
        band[:,k1,k2], eigwf[:,:,k1,k2] = eigen(Hermitian(H[k1,k2,:,:]) )
        eigwf[:,1,k1,k2] *= cis(-angle(eigwf[1,1,k1,k2]))
        eigwf[:,2,k1,k2] *= cis(-angle(eigwf[1,2,k1,k2]))
    end
    return LLTDHFGroundState(HFpara=HFpara, band=band, eigwf=eigwf)
end


"""
Input the excitation momentum q=(q1, q2) and the numerical parameters
q1, q2 can be non-integers in units of 1/N1, 1/N2
Output the interaction coefficient in form of
V[s1, s2, s3, s4, p1, p2, k1, k2]
for noninteger q1, q2, V has extra two dimensions for p_shift and k_shift
V[s1, s2, s3, s4, p1, p2, k1, k2, p_shift, k_shift]
"""
function TDHF_V_matrix_spin(q1::Int64, q2::Int64, HFpara::LLHFNumPara; Gshell = 2)
    N1 = HFpara.N1; N2 = HFpara.N2;
    LL = HFpara.LL
    sys = HFpara.system 
    G1 = sys.G1; G2 = sys.G2
    Vspin = zeros(ComplexF64,2,2,2,2,N1,N2,N1,N2)

    # Hartree
    g1_shift_hartree = div(q1, N1, RoundNearest)
    g2_shift_hartree = div(q2, N2, RoundNearest)
    for g1 in -Gshell:Gshell, g2 in -Gshell:Gshell
        if abs(g1+g2)>Gshell
            continue
        end
        g1 -= g1_shift_hartree
        g2 -= g2_shift_hartree
        qg1 = q1 + g1 * N1
        qg2 = q2 + g2 * N2
        if qg1==0 && qg2==0
            continue
        end
        V = V_int(qg1, qg2; N1=N1, N2=N2, r12 = sys.ratio12,
            Gl=sys.Gl, D_l=sys.D/sys.l, cosθ=sys.cosθ
        )

        for s12 = [1;-1], s34 = [1;-1]
            VFF = Form_factor(LL,LL, (-qg1*G1/N1-qg2*G2/N2)..., s12, sys.l) * 
                Form_factor(LL,LL, (qg1*G1/N1+qg2*G2/N2)..., s34, sys.l) * V
            if isnan(HFpara.λ)
                s12 != s34 && (VFF *= HFpara.α)
            elseif isnan(HFpara.α)
                VFF *= HFpara.λ
            end
            for p1 in 0:N1-1, p2 in 0:N2-1, k1 in 0:N1-1, k2 in 0:N2-1
                Vspin[(3-s12)÷2, (3-s12)÷2, (3-s34)÷2, (3-s34)÷2, 
                    1+p1, 1+p2, 1+k1, 1+k2] += VFF * 
                cis(ql_cross((s12*p1-s34*k1)/N1, (s12*p2-s34*k2)/N2, g1, g2)
                    + 0.5s12*ql_cross(q1/N1, q2/N2, -p1/N1-g1, -p2/N2-g2)
                    + 0.5s34*ql_cross(q1/N1, q2/N2,  k1/N1-g1,  k2/N2-g2)
                )
            end
        end
    end

    # Fock
    Threads.@threads for Ipk in CartesianIndices(axes(Vspin)[5:8])
        p1, p2, k1, k2 = Tuple(Ipk) .- 1
        kpq1 = k1 - p1 + q1
        kpq2 = k2 - p2 + q2
        g1_shift = div(kpq1, N1, RoundNearest)
        g2_shift = div(kpq2, N2, RoundNearest)
        for g1 in -Gshell:Gshell, g2 in -Gshell:Gshell
            if abs(g1+g2)>Gshell
                continue
            end
            g1 -= g1_shift
            g2 -= g2_shift
            qg1 = kpq1 + g1 * N1
            qg2 = kpq2 + g2 * N2
            V = V_int(qg1, qg2; N1=N1, N2=N2, r12=sys.ratio12, 
                Gl=sys.Gl, D_l=sys.D/sys.l, cosθ=sys.cosθ
            )

            phase_angle1 = (ql_cross(p1/N1, p2/N2, k1/N1, k2/N2) +
                ql_cross((k1+p1)/N1, (k2+p2)/N2, g1, g2)
            )
            phase_angle2 = ql_cross(q1/N1, q2/N2,  p1/N1-g1,  p2/N2-g2)
            phase_angle3 = ql_cross(q1/N1, q2/N2, -k1/N1-g1, -k2/N2-g2)


            for s14 = [1;-1], s23 = [1;-1]

                VFF = Form_factor(LL,LL, (-qg1*G1/N1-qg2*G2/N2)..., s14, sys.l) * 
                    Form_factor(LL,LL, (qg1*G1/N1+qg2*G2/N2)..., s23, sys.l) * V
                if isnan(HFpara.λ)
                    s14 != s23 && (VFF *= HFpara.α)
                elseif isnan(HFpara.α)
                    s14 == s23 && (VFF *= HFpara.λ)
                end
                Vspin[(3-s14)÷2,(3-s23)÷2,(3-s23)÷2,(3-s14)÷2,Ipk] -= VFF * cis(
                    0.5*(s14-s23) * phase_angle1 +
                    0.5s23 * phase_angle2 +
                    0.5s14 * phase_angle3
                )
            end
        end
    end
    return Vspin
end
function TDHF_V_matrix_spin(q1, q2, HFpara::LLHFNumPara; Gshell = 2)
    N1 = HFpara.N1; N2 = HFpara.N2;
    LL = HFpara.LL
    sys = HFpara.system 
    G1 = sys.G1; G2 = sys.G2
    # add extra two dimensions for non-int q shift
    Vspin = zeros(ComplexF64,2,2,2,2,N1,N2,N1,N2, 2, 2)

    # Hartree
    g1_shift = div(q1, N1, RoundNearest)
    g2_shift = div(q2, N2, RoundNearest)
    for g1 in -Gshell:Gshell, g2 in -Gshell:Gshell
        if abs(g1+g2)>Gshell
            continue
        end
        g1 -= g1_shift
        g2 -= g2_shift
        qg1 = q1 + g1 * N1
        qg2 = q2 + g2 * N2
        if qg1==0 && qg2==0
            continue
        end
        V = V_int(qg1, qg2; N1=N1, N2=N2, r12 = sys.ratio12,
            Gl=sys.Gl, D_l=sys.D/sys.l, cosθ=sys.cosθ
        )

        for s12 = [1;-1], s34 = [1;-1]
            phase = [ begin
                    k1 -= q1*k_shift; k2 -= q2*k_shift;
                    p1 += q1*p_shift; p2 += q2*p_shift;
                    cis(ql_cross((s12*p1-s34*k1)/N1, (s12*p2-s34*k2)/N2, g1, g2)
                    + 0.5s12*ql_cross(q1/N1, q2/N2, -p1/N1-g1, -p2/N2-g2)
                    + 0.5s34*ql_cross(q1/N1, q2/N2,  k1/N1-g1,  k2/N2-g2)) 
                end
                for p1 in 0:N1-1, p2 in 0:N2-1, k1 in 0:N1-1, k2 in 0:N2-1,
                    p_shift in 0:1, k_shift in 0:1
            ]
            VFF = Form_factor(LL,LL, (-qg1*G1/N1-qg2*G2/N2)..., s12, sys.l) * 
                Form_factor(LL,LL, (qg1*G1/N1+qg2*G2/N2)..., s34, sys.l) * V
            if isnan(HFpara.λ)
                s12 != s34 && (VFF *= HFpara.α)
            elseif isnan(HFpara.α)
                VFF *= HFpara.λ
            end
            Vspin[(3-s12)÷2, (3-s12)÷2, (3-s34)÷2, (3-s34)÷2,
                :,:,:,:,:,:] .+= VFF .* phase 
        end
    end

    # Fock
    for Ipk in CartesianIndices(axes(Vspin)[5:10])
        p1, p2, k1, k2, p_shift, k_shift = Tuple(Ipk) .- 1
        k1 -= q1 * k_shift
        k2 -= q2 * k_shift
        p1 += q1 * p_shift
        p2 += q2 * p_shift
        kpq1 = k1 - p1 + q1
        kpq2 = k2 - p2 + q2
        g1_shift = div(kpq1, N1, RoundNearest)
        g2_shift = div(kpq2, N2, RoundNearest)
        for g1 in -Gshell:Gshell, g2 in -Gshell:Gshell
            if abs(g1+g2)>Gshell
                continue
            end
            g1 -= g1_shift
            g2 -= g2_shift
            qg1 = kpq1 + g1 * N1
            qg2 = kpq2 + g2 * N2
            V = V_int(qg1, qg2; N1=N1, N2=N2, r12=sys.ratio12, 
                Gl=sys.Gl, D_l=sys.D/sys.l, cosθ=sys.cosθ
            )

            phase_angle1 = (ql_cross(p1/N1, p2/N2, k1/N1, k2/N2) +
                ql_cross((k1+p1)/N1, (k2+p2)/N2, g1, g2)
            )
            phase_angle2 = ql_cross(q1/N1, q2/N2,  p1/N1-g1,  p2/N2-g2)
            phase_angle3 = ql_cross(q1/N1, q2/N2, -k1/N1-g1, -k2/N2-g2)


            for s14 = [1;-1], s23 = [1;-1]
                phase = cis(
                    0.5*(s14-s23) * phase_angle1 +
                    0.5s23 * phase_angle2 +
                    0.5s14 * phase_angle3
                )
                VFF = Form_factor(LL,LL, (-qg1*G1/N1-qg2*G2/N2)..., s14, sys.l) * 
                    Form_factor(LL,LL, (qg1*G1/N1+qg2*G2/N2)..., s23, sys.l) * V
                if isnan(HFpara.λ)
                    s14 != s23 && (VFF *= HFpara.α)
                elseif isnan(HFpara.α)
                    s14 == s23 && (VFF *= HFpara.λ)
                end
                Vspin[(3-s14)÷2,(3-s23)÷2,(3-s23)÷2,(3-s14)÷2,
                    Ipk] -= phase * VFF
            end
        end
    end
    return Vspin
end


# interpolation of triangular lattices
# only work for cosθ = 1/2
function band_ipl(GS::LLTDHFGroundState, τ, k1, k2)

    if abs(GS.HFpara.system.cosθ - 0.5) > 1e-5
        error("the interpolation is designed for non-integer q and only works for triangular lattice with cosθ=1/2")
    end
    N1 = GS.HFpara.N1
    N2 = GS.HFpara.N2

    k1_int = floor(Int64, k1)
    k1_frac = k1 - k1_int
    k1_int = mod1(k1_int, N1)
    k1_int_1 = mod1(k1_int+1, N1)

    k2_int = floor(Int64, k2)
    k2_frac = k2 - k2_int
    k2_int = mod1(k2_int, N2)
    k2_int_1 = mod1(k2_int+1, N2)

    x = 0.0
    if k1_frac + k2_frac <= 1.
        x += GS.band[τ, k1_int, k2_int] * (1 - k1_frac - k2_frac)
        x += GS.band[τ, k1_int_1, k2_int] * k1_frac
        x += GS.band[τ, k1_int, k2_int_1] * k2_frac
    else
        x += GS.band[τ, k1_int_1, k2_int_1] * (k1_frac + k2_frac - 1)
        x += GS.band[τ, k1_int_1, k2_int] * (1 - k2_frac)
        x += GS.band[τ, k1_int, k2_int_1] * (1 - k1_frac)
    end
    return x
end



"""
Matrix ZE+XV is the Hessian of the energy functional
should be positive semidefinite
``
V^{\textbf{p}-\textbf{q} s_1} {}_{\textbf{p} s_2,} 
{}^{\textbf{k}+\textbf{q} s_3} {}_{\textbf{k} s_4}
→
V^{\textbf{p}-\textbf{q} n_1} {}_{\textbf{p} n_2,} 
{}^{\textbf{k}+\textbf{q} n_3} {}_{\textbf{k} n_4}
``
"""
function TDHF_ZEXV(q1::Int64, q2::Int64, GS::LLTDHFGroundState; 
    Vspin = TDHF_V_matrix_spin(q1, q2, GS.HFpara))

    N1 = GS.HFpara.N1; N2 = GS.HFpara.N2
    band = GS.band; cov_s2n = GS.eigwf; ctr_s2n = conj(cov_s2n)
    @inline fld0mod1(x,y) = fldmod1(x,y) .- (1, 0)

    M = zeros(ComplexF64, N1, N2, 2, N1, N2, 2)
    # ZE
    for k2 in axes(M,2), k1 in axes(M,1)
        M[k1,k2,1,k1,k2,1] += band[2,mod1(k1+q1,N1),mod1(k2+q2,N2)] - band[1,k1,k2]
        M[k1,k2,2,k1,k2,2] += band[2,mod1(k1-q1,N1),mod1(k2-q2,N2)] - band[1,k1,k2]
    end
    Vspin *= GS.HFpara.system.W0 / GS.HFpara.k_num
    # XV
    for (k1, k2) in Iterators.product(1:N1, 1:N2)
        kG1, kq1 = fld0mod1(k1-q1, N1)
        kG2, kq2 = fld0mod1(k2-q2, N2)
        k_phase = -0.5ql_cross(kG1, kG2, (2(k1-1)-q1)/N1, (2(k2-1)-q2)/N2)
        for (p1, p2) in Iterators.product(1:N1, 1:N2)
            pG1, pq1 = fld0mod1(p1+q1, N1)
            pG2, pq2 = fld0mod1(p2+q2, N2)
            p_phase = -0.5ql_cross(pG1, pG2, (2(p1-1)+q1)/N1, (2(p2-1)+q2)/N2)
            for (s1,s2,s3,s4) in Iterators.product(1:2,1:2,1:2,1:2)
                # M[cv,cv]
                M[p1, p2, 2, k1, k2, 1] += Vspin[s1,s2,s3,s4,p1,p2,k1,k2] *
                ctr_s2n[s1,2,p1,p2]*cov_s2n[s2,1,p1,p2]*ctr_s2n[s3,2,k1,k2]*cov_s2n[s4,1,k1,k2]
                #(s1==2 && s2==1 && s3==2 && s4==1)
                # M[vc,cv]
                M[p1, p2, 1, k1, k2, 1] += Vspin[s1,s2,s3,s4,pq1,pq2,k1,k2] *
                ctr_s2n[s1,1,p1,p2]*cov_s2n[s2,2,p1,p2]*ctr_s2n[s3,2,k1,k2]*cov_s2n[s4,1,k1,k2] *
                #(s1==1 && s2==2 && s3==2 && s4==1) *
                cis((s1 != s2) * (3-2s1) * p_phase)
                # M[cv,vc]
                M[p1, p2, 2, k1, k2, 2] += Vspin[s1,s2,s3,s4,p1,p2,kq1,kq2] *
                ctr_s2n[s1,2,p1,p2]*cov_s2n[s2,1,p1,p2]*ctr_s2n[s3,1,k1,k2]*cov_s2n[s4,2,k1,k2] *
                #(s1==2 && s2==1 && s3==1 && s4==2) *
                cis((s3 != s4) * (3-2s3) * k_phase)
                # M[vc,vc]
                M[p1, p2, 1, k1, k2, 2] += Vspin[s1,s2,s3,s4,pq1,pq2,kq1,kq2] *
                ctr_s2n[s1,1,p1,p2]*cov_s2n[s2,2,p1,p2]*ctr_s2n[s3,1,k1,k2]*cov_s2n[s4,2,k1,k2] *
                #(s1==1 && s2==2 && s3==1 && s4==2) *
                cis((s1 != s2) * (3-2s1) * p_phase + (s3 != s4) * (3-2s3) * k_phase)
            end
        end
    end
    return M
end
function TDHF_ZEXV(q1, q2, GS::LLTDHFGroundState; 
    Vspin = TDHF_V_matrix_spin(q1, q2, GS.HFpara))

    N1 = GS.HFpara.N1; N2 = GS.HFpara.N2
    band = GS.band; cov_s2n = GS.eigwf; ctr_s2n = conj(cov_s2n)

    M = zeros(ComplexF64, N1, N2, 2, N1, N2, 2)
    # ZE
    for k2 in axes(M,2), k1 in axes(M,1)
        M[k1,k2,1,k1,k2,1] += band_ipl(GS, 2, k1+q1, k2+q2) - band[1,k1,k2]
        M[k1,k2,2,k1,k2,2] += band_ipl(GS, 2, k1-q1, k2-q2) - band[1,k1,k2]
    end
    Vspin *= GS.HFpara.system.W0 / GS.HFpara.k_num
    # XV
    for (k1, k2) in Iterators.product(1:N1, 1:N2)
        for (p1, p2) in Iterators.product(1:N1, 1:N2)
            for (s1,s2,s3,s4) in Iterators.product(1:2,1:2,1:2,1:2)
                # M[cv,cv]
                M[p1, p2, 2, k1, k2, 1] += Vspin[s1,s2,s3,s4,p1,p2,k1,k2,1,1] *
                ctr_s2n[s1,2,p1,p2]*cov_s2n[s2,1,p1,p2]*ctr_s2n[s3,2,k1,k2]*cov_s2n[s4,1,k1,k2]
                #(s1==2 && s2==1 && s3==2 && s4==1)
                # M[vc,cv]
                M[p1, p2, 1, k1, k2, 1] += Vspin[s1,s2,s3,s4,p1,p2,k1,k2,2,1] *
                ctr_s2n[s1,1,p1,p2]*cov_s2n[s2,2,p1,p2]*ctr_s2n[s3,2,k1,k2]*cov_s2n[s4,1,k1,k2]
                #(s1==1 && s2==2 && s3==2 && s4==1)
                # M[cv,vc]
                M[p1, p2, 2, k1, k2, 2] += Vspin[s1,s2,s3,s4,p1,p2,k1,k2,1,2] *
                ctr_s2n[s1,2,p1,p2]*cov_s2n[s2,1,p1,p2]*ctr_s2n[s3,1,k1,k2]*cov_s2n[s4,2,k1,k2]
                #(s1==2 && s2==1 && s3==1 && s4==2)
                # M[vc,vc]
                M[p1, p2, 1, k1, k2, 2] += Vspin[s1,s2,s3,s4,p1,p2,k1,k2,2,2] *
                ctr_s2n[s1,1,p1,p2]*cov_s2n[s2,2,p1,p2]*ctr_s2n[s3,1,k1,k2]*cov_s2n[s4,2,k1,k2]
                #(s1==1 && s2==2 && s3==1 && s4==2)
            end
        end
    end
    return M
end
#=
    # depreciated
    function TDHF_ZEXV(q1::Int64, q2::Int64, GS::LLTDHFGroundState; 
        Vspin = TDHF_Vspin(q1, q2, GS.HFpara))
        N1 = GS.HFpara.N1; N2 = GS.HFpara.N2
        band = GS.band; cov_s2n = GS.eigwf; ctr_s2n = conj(cov_s2n)
        @inline fld0mod1(x,y) = fldmod1(x,y) .- (1, 0)
        M = zeros(ComplexF64, N1, N2, 2, N1, N2, 2)
        # ZE
        for k2 in axes(M,2), k1 in axes(M,1)
            M[k1,k2,1,k1,k2,1] += band[2,mod1(k1+q1,N1),mod1(k2+q2,N2)] - band[1,k1,k2]
            M[k1,k2,2,k1,k2,2] += band[2,mod1(k1-q1,N1),mod1(k2-q2,N2)] - band[1,k1,k2]
        end
        Vspin *= GS.HFpara.system.W0 / GS.HFpara.k_num
        # XV
        for (k1, k2) in Iterators.product(1:N1, 1:N2)
            kG1, kq1 = fld0mod1(k1+q1, N1)
            kG2, kq2 = fld0mod1(k2+q2, N2)
            k_phase = 0.5ql_cross(kG1, kG2, (2(k1-1)+q1)/N1, (2(k2-1)+q2)/N2)
            #println(k1, "\t", k2, "\t", kq1, "\t", kq2, "\t", kG1, "\t", kG2)
            for (p1, p2) in Iterators.product(1:N1, 1:N2)
                pG1, pq1 = fld0mod1(p1-q1, N1)
                pG2, pq2 = fld0mod1(p2-q2, N2)
                p_phase = 0.5ql_cross(pG1, pG2, (2(p1-1)-q1)/N1, (2(p2-1)-q2)/N2)
                for (s1,s2,s3,s4) in Iterators.product(1:2,1:2,1:2,1:2)
                    # M[cv,cv]
                    M[p1, p2, 2, k1, k2, 1] += Vspin[s1,s2,s3,s4,p1,p2,k1,k2] *
                    ctr_s2n[s1,2,p1,p2]*cov_s2n[s2,1,p1,p2]*ctr_s2n[s3,2,k1,k2]*cov_s2n[s4,1,k1,k2]
                    #(s1==2 && s2==1 && s3==2 && s4==1)
                    # M[vc,cv]
                    M[pq1, pq2, 1, k1, k2, 1] += Vspin[s1,s2,s3,s4,p1,p2,k1,k2] *
                    ctr_s2n[s1,1,pq1,pq2]*cov_s2n[s2,2,pq1,pq2]*ctr_s2n[s3,2,k1,k2]*cov_s2n[s4,1,k1,k2] *
                    #(s1==1 && s2==2 && s3==2 && s4==1) *
                    cis((s1 != s2) * (3-2s1) * p_phase)
                    # M[cv,vc]
                    M[p1, p2, 2, kq1, kq2, 2] += Vspin[s1,s2,s3,s4,p1,p2,k1,k2] *
                    ctr_s2n[s1,2,p1,p2]*cov_s2n[s2,1,p1,p2]*ctr_s2n[s3,1,kq1,kq2]*cov_s2n[s4,2,kq1,kq2] *
                    #(s1==2 && s2==1 && s3==1 && s4==2) *
                    cis((s3 != s4) * (3-2s3) * k_phase)
                    # M[vc,vc]
                    M[pq1, pq2, 1, kq1, kq2, 2] += Vspin[s1,s2,s3,s4,p1,p2,k1,k2] *
                    ctr_s2n[s1,1,pq1,pq2]*cov_s2n[s2,2,pq1,pq2]*ctr_s2n[s3,1,kq1,kq2]*cov_s2n[s4,2,kq1,kq2] *
                    #(s1==1 && s2==2 && s3==1 && s4==2) *
                    cis((s1 != s2) * (3-2s1) * p_phase + (s3 != s4) * (3-2s3) * k_phase)
                end
            end
        end
        return M
    end
=#


function TDHF_solve(ZEXV, n = 4)
    M = copy(ZEXV)
    M[:,:,2,:,:,:] .*= -1
    N = size(M,1)*size(M,2)*2
    M = reshape(M, (N,N))
    vec0 = rand(ComplexF64, N)
    vals, vecs, info = eigsolve(M, vec0, 2n, EigSorter(abs,rev=false), ishermitian=false);
    return vals[1:2n]#, vecs[1:2n]
    #perm = sortperm(real.(vals))[n+1:2n]
    #return vals[perm], vecs[perm]
end

end




# end