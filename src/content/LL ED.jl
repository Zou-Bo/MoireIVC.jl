"""
Do Exact Diagonalization for two regions in BZ.
The two regions contains two Dirac Points with large momentum kD1, kD2.
Small momentum is the momentum relative to Dirac Points (DP)
Use Landau level quasi-Bloch wavefunctions

Import same system parameters from LLHF.

Input numerical parameter includes:
Some k points around the two Dirac points,
One-body Hamiltonian determined by outside mean field.
Valley number is not a good quantum number.
Without Umklapp, electron number in each DP region is conserved.

Calculate the ED within the block specified with:
electron number in DC and total small momentum.

"""
module LLED

export LLEDPara, LLED_init
export MBS, LLED_mbslist_twoDiracCone
export LLED_block_bysmallmomentum, LLED_interactionlist
export LLED_Block_Hamiltonian, LLED_solve
public reduced_density_matrix, one_body_reduced_density_matrix, entanglement_entropy 




using MKL, LinearAlgebra
using PhysicalUnits
using Combinatorics
using ExtendableSparse, KrylovKit
using MoireIVC.Basics: laguerrel, ql_cross, my_searchsortedfirst
using MoireIVC.LLHF: LLHFSysPara, LLHFNumPara

# initiation
begin
    @kwdef struct LLEDPara
        k_list::Matrix{Int64}
        k_num::Int64 = size(k_list, 2)

        # H_meanfield[k,s,s',kD] c†_{k,s,kD} c_{k,s',kD}
        Hmf::Array{ComplexF64, 4}

        # Nk × Nk k-mesh, need to be a multiple of 3
        Nk::Int64
        LL::Int64
        system::LLHFSysPara
    end
    """
    parameters of ED calculation:
    Nk×Nk k-mesh where 3|Nk.
    LL -> Landau level index.
    k_list[:,k] = [k1; k2] -> "small" momentum = k1/Nk * G1 +  k2/Nk * G2,
    where k: 1~k_num.
    Dirac point "large" momentum: kD1 = 1/3 * (G1+G2), kD2 = 2/3 * (G1+G2).
    mean-field Hamiltonian: Hmf[k,s,s',kD] c†_{k,s,kD} c_{k,s',kD}.
    system::LLHFSysPara contains G1, G2, Gl.
    """
    function LLED_init(k_list, Hmf, syspara, N, LL)
        if N%3 !=0
            N = N%3 * 3
            @warn "Input k-mesh is not 3n×3n, changed to $N×$N."
        end
        k_num = size(k_list, 2)
        if size(k_list, 1) != 2
            error("Input k_list should have 2 rows")
        end
        if size(Hmf) != (k_num, 2, 2, 2)
            error("Input mean field size incompatible to k_list")
        end
        return LLEDPara(; k_list = k_list, Hmf = Hmf, Nk = N, system = syspara, LL = LL)
    end
end




##################################################
# Single electron states are labeled with 
# 1. spin: s=1,2 for spin up and down
# 2. Dirac point: kD = 1,2 
# 3. small momentum: k = 1~k_num in the input liste
# The many-body state denoted by Int64/bit-array
# Each bit is a single-electron state 1 << i
# Bit index i = (k-1) + k_num(s-1) + 2k_num(kD-1)
##################################################
begin
    """
    The struct of many-body state consists of 
        n: the integer consist of bit array
        i: vector of occupies bit numbers
        k1, k2: total small momentum
    """
    struct MBS
        n::Int64
        i::Vector{Int64}
        k1::Int64
        k2::Int64
        bits::Int64
    end
    import Base: *, isless
    function *(mbs1::MBS, mbs2::MBS)
        MBS(mbs1.n << mbs2.bits | mbs2.n, vcat(mbs2.i, (mbs1.i) .+ mbs2.bits ),
            mbs1.k1 + mbs2.k1, mbs1.k2 + mbs2.k2, mbs1.bits+mbs2.bits
        )
    end
    function isless(mbs1::MBS, mbs2::MBS)
        isless(mbs1.n, mbs2.n)
    end


    """
    Return the total small momentum K1 and K2 of a many body state.
    bit index i = (k-1) + k_num(s-1) + 2k_num(kD-1)
    """
    function TotalMomentum(para::LLEDPara, i_list::Vector{Int})
        k1 = 0; k2 = 0
        for i in i_list
            momentum = para.k_list[:, i%para.k_num + 1]
            k1 += momentum[1]
            k2 += momentum[2]
        end
        return k1, k2
    end


    function mbslist_oneDiracCone(para::LLEDPara, N_in_one::Int64)
        k_num = para.k_num
        N_in_one > 2k_num && error("the maximum number of electrons
            in one Dirac cone is $(2k_num) set by current k_index_list")
        sort([MBS(sum(1 .<< combi), combi, TotalMomentum(para,combi)..., 2k_num)
            for combi in collect(combinations(0:2k_num-1, N_in_one))
        ])
    end
    """
    define DiracCone distribution
    return a list of MBS with electron number (N1, N2)
    """
    LLED_mbslist_twoDiracCone(para::LLEDPara, N1::Int64, N2::Int64) = 
        kron(mbslist_oneDiracCone(para, N2), mbslist_oneDiracCone(para, N1))
    """
    return blocks with same samll total momentum,
    Input: MBS list
    Output:
    1. list of block MBS list,
    2. list of k1 of blocks,
    3. list of k2 if blocks,
    4. list incex of the k1=k2=0 block (return 0 if not exist)
    """
    function LLED_block_bysmallmomentum(mbs_list::Vector{MBS};
        momentum_restriction = true, k1range=(-2,2), k2range=(-2,2))

        blocks = Vector{MBS}[];
        block_k1 = Int64[];
        block_k2 = Int64[];
        k1min, k1max = extrema(getfield.(mbs_list,:k1))
        k2min, k2max = extrema(getfield.(mbs_list,:k2))
        if momentum_restriction
            k1min = max(k1range[1], k1min)
            k2min = max(k2range[1], k2min)
            k1max = min(k1range[2], k1max)
            k2max = min(k2range[2], k2max)
        end
        for K1 in k1min:k1max, K2 in k2min:k2max
            mask = findall(mbs -> mbs.k1==K1 && mbs.k2==K2, mbs_list)
            if length(mask) != 0
                push!(blocks, mbs_list[mask])
                push!(block_k1, K1)
                push!(block_k2, K2)
            end
        end
        block_num_0 = findfirst(eachindex(block_k1)) do bn
            block_k1[bn]==0 && block_k2[bn]==0
        end
        return blocks, block_k1, block_k2, block_num_0
    end
end




# generate interaction terms (V, i4, i3, i2, i1)
begin
    "calculate interaction strength based on (q1, q2)*G/Nk"
    function VFF(q::Vector{Int64}, para::LLEDPara)
        q1, q2 = q
        Gl = para.system.Gl
        W0 = para.system.W0
        D_l = para.system.D / para.system.l
        Nk = para.Nk
        LL = para.LL

        if q1 == 0 && q2 == 0
            return D_l * W0 / Nk^2
        else
            ql = sqrt(q1^2 + q2^2 + q1*q2) * Gl / Nk
            V = 1.0 / ql * tanh(ql*D_l)
            FF = laguerrel(LL, 0, ql^2/2)^2 * exp(-ql^2/2)
            return V * FF * W0 / Nk^2
        end
    end
    """
    calculate the interaction phase: 

    <kf | exp{-i[q⋅r + G⋅r]} | ki>, q=ki-kf

    For <kf | exp{+i[q⋅r + G⋅r]} | ki>, q=kf-ki,
    use -G simply.
    """
    function Form_phase(ki, kf, G, spin, Nk)
        ki1 = ki[1]; ki2 = ki[2]
        kf1 = kf[1]; kf2 = kf[2]
        phase = 0.5*ql_cross(kf1/Nk, kf2/Nk, ki1/Nk, ki2/Nk)
        phase+= 0.5*ql_cross((ki1+kf1)/Nk, (ki2+kf2)/Nk, G[1], G[2])
        phase *= 3-2spin
        return phase
    end
    # find groups of pairs
    function k_pair_groups(para::LLEDPara)
        k_num = para.k_num
        pair_groups = Vector{Vector{Int64}}[]
        k_pairs = [MBS(0, pair, TotalMomentum(para, pair)..., 0)
            for pair in collect(with_replacement_combinations(0:k_num-1, 2))
        ]
        k1min, k1max = extrema(mbs -> mbs.k1, k_pairs)
        k2min, k2max = extrema(mbs -> mbs.k2, k_pairs)
        for K1 in k1min:k1max, K2 in k2min:k2max
            group = findall(mbs -> mbs.k1==K1 && mbs.k2==K2, k_pairs)
            if length(group) != 0
                push!(pair_groups, getfield.(k_pairs[group] ,:i))
            end
        end
        return pair_groups
    end
    # normal order
    function normal_order_push!(int_list, int)

        V, i4, i3, i2, i1 = int

        if i4 == i3 && i1 == i2
            return
        end
        if i4 < i3
            i3, i4 = (i4, i3)
            V = -V
        end
        if i1 < i2
            i1, i2 = (i2, i1)
            V = -V
        end
        if i1 < i4
            i1, i4 = (i4, i1)
            i2, i3 = (i3, i2)
            V = conj(V)
        end
        push!(int_list, (V, i4, i3, i2, i1))
    end
    """
    generate interaction terms
    step 1:
    Pair the small momentum, and 
    divide them into groups with same total small momentum
    The scattering conserved small momentum,
    incident pair (1,2) -> output pair (3,4) have same samll momentum
    step 2:
    list all the scattering term in form of:
    V * c†_4 c†_3 c_2 c_1 (+ h.c. if 1,2≠3,4)
    use normal order: i4 > i3 && i1 > i2 && i1 >= i4
    explain:
    (1) switch 3<->4 and 1<->2 gives the same term, canceling off 1/2 prefactor
    (2) switch only 3<->4 corresponds to the Hartree and Fock separately, summed in amplitute V.
    (3) switch 1<->4 and 2<->3 gives complex conjugate:
    only the normal ordered one is listed here,
    the other will be considered when generating H.
    """
    function LLED_interactionlist(para::LLEDPara; groups = k_pair_groups(para), g_shell = 3)
        
        k_list = para.k_list
        k_num = para.k_num
        NK = para.Nk÷3

        list = Tuple{ComplexF64, Int64, Int64, Int64, Int64}[]

        for group in groups
            for pair_in in eachindex(group), pair_out in eachindex(group)
                
                # scattering from pair_in to pair_out
                if pair_out > pair_in
                    continue
                end
                k2, k1 = sort(group[pair_in ] .+1 )
                k3, k4 = sort(group[pair_out] .+1 )


                # 1. same kD
                for kD in 1:2
                    k1index = k_list[:,k1] + [kD*NK; kD*NK]
                    k2index = k_list[:,k2] + [kD*NK; kD*NK]
                    k3index = k_list[:,k3] + [kD*NK; kD*NK]
                    k4index = k_list[:,k4] + [kD*NK; kD*NK]
                    
                    #1.1 same spin
                    for spin in 1:2
                        i1 = k1-1 + k_num*(spin-1) + 2k_num*(kD-1)
                        i2 = k2-1 + k_num*(spin-1) + 2k_num*(kD-1)
                        i3 = k3-1 + k_num*(spin-1) + 2k_num*(kD-1)
                        i4 = k4-1 + k_num*(spin-1) + 2k_num*(kD-1)
                        if i1 == i2 || i3 == i4
                            continue
                        end

                        V = ComplexF64(0.0)
                        for g1 in -g_shell:g_shell, g2 in -g_shell:g_shell
                            if abs(g1+g2) > g_shell
                                continue
                            end
                            V += VFF(k4index .- k1index .+ (g1, g2).*3NK, para) * 
                                cis(Form_phase(k1index, k4index, (g1,g2), spin, 3NK)
                                + Form_phase(k2index, k3index, (-g1,-g2), spin, 3NK))
                            V -= VFF(k3index .- k1index .+ (g1, g2).*3NK, para) * 
                                cis(Form_phase(k1index, k3index, (g1,g2), spin, 3NK)
                                + Form_phase(k2index, k4index, (-g1,-g2), spin, 3NK))
                        end
                        
                        normal_order_push!(list, (V,i4,i3,i2,i1))
                    end

                    # 1.2 opposite spin
                    for spin1 in 1:2, spin4 in 1:2
                        spin2 = 3-spin1;
                        spin3 = 3-spin4;
                        if k1==k2 && spin1 < spin2 || k3==k4 && spin4 < spin3
                            continue
                        end
                        if pair_in == pair_out && spin1 < spin4
                            continue
                        end
                        i1 = k1-1 + k_num*(spin1-1) + 2k_num*(kD-1)
                        i2 = k2-1 + k_num*(spin2-1) + 2k_num*(kD-1)
                        i3 = k3-1 + k_num*(spin3-1) + 2k_num*(kD-1)
                        i4 = k4-1 + k_num*(spin4-1) + 2k_num*(kD-1)


                        V = ComplexF64(0.0)
                        for g1 in -g_shell:g_shell, g2 in -g_shell:g_shell
                            if abs(g1+g2) > g_shell
                                continue
                            end
                            if spin1 == spin4
                                V += VFF(k4index .- k1index .+ (g1, g2).*3NK, para) * 
                                cis(Form_phase(k1index, k4index, (g1,g2), spin1, 3NK)
                                + Form_phase(k2index, k3index, (-g1,-g2), spin2, 3NK))
                            else
                                V -= VFF(k3index .- k1index .+ (g1, g2).*3NK, para) * 
                                cis(Form_phase(k1index, k3index, (g1,g2), spin1, 3NK)
                                + Form_phase(k2index, k4index, (-g1,-g2), spin2, 3NK))
                            end
                        end

                        normal_order_push!(list, (V,i4,i3,i2,i1))
                    end
                end

                
                # 2. opposite kD
                for kD1 in 1:2, kD4 in 1:2
                    kD2 = 3-kD1;
                    kD3 = 3-kD4;
                    if k1==k2 && kD1 < kD2 || k3==k4 && kD4 < kD3
                        continue
                    end
                    k1index = k_list[:,k1] + [kD1*NK; kD1*NK]
                    k2index = k_list[:,k2] + [kD2*NK; kD2*NK]
                    k3index = k_list[:,k3] + [kD3*NK; kD3*NK]
                    k4index = k_list[:,k4] + [kD4*NK; kD4*NK]
                    
                    #2.1 same spin
                    for spin in 1:2
                        if pair_in == pair_out && kD1 < kD4
                            continue
                        end
                        i1 = k1-1 + k_num*(spin-1) + 2k_num*(kD1-1)
                        i2 = k2-1 + k_num*(spin-1) + 2k_num*(kD2-1)
                        i3 = k3-1 + k_num*(spin-1) + 2k_num*(kD3-1)
                        i4 = k4-1 + k_num*(spin-1) + 2k_num*(kD4-1)

                        V = ComplexF64(0.0)
                        for g1 in -g_shell:g_shell, g2 in -g_shell:g_shell
                            if abs(g1+g2) > g_shell
                                continue
                            end
                            V += VFF(k4index .- k1index .+ (g1, g2).*3NK, para) * 
                                cis(Form_phase(k1index, k4index, (g1,g2), spin, 3NK)
                                + Form_phase(k2index, k3index, (-g1,-g2), spin, 3NK))
                            V -= VFF(k3index .- k1index .+ (g1, g2).*3NK, para) * 
                                cis(Form_phase(k1index, k3index, (g1,g2), spin, 3NK)
                                + Form_phase(k2index, k4index, (-g1,-g2), spin, 3NK))
                        end

                        normal_order_push!(list, (V,i4,i3,i2,i1))
                    end

                    # 2.2 opposite spin
                    for spin1 in 1:2, spin4 in 1:2
                        spin2 = 3-spin1;
                        spin3 = 3-spin4;
                        if pair_in == pair_out 
                            if spin1==spin4 && kD1 < kD4
                                continue
                            elseif kD1==kD4 && spin1 < spin4
                                continue
                            end
                        end
                        i1 = k1-1 + k_num*(spin1-1) + 2k_num*(kD1-1)
                        i2 = k2-1 + k_num*(spin2-1) + 2k_num*(kD2-1)
                        i3 = k3-1 + k_num*(spin3-1) + 2k_num*(kD3-1)
                        i4 = k4-1 + k_num*(spin4-1) + 2k_num*(kD4-1)


                        V = ComplexF64(0.0)
                        for g1 in -g_shell:g_shell, g2 in -g_shell:g_shell
                            if abs(g1+g2) > g_shell
                                continue
                            end
                            if spin1 == spin4
                                V += VFF(k4index .- k1index .+ (g1, g2).*3NK, para) * 
                                cis(Form_phase(k1index, k4index, (g1,g2), spin1, 3NK)
                                + Form_phase(k2index, k3index, (-g1,-g2), spin2, 3NK))
                            else
                                V -= VFF(k3index .- k1index .+ (g1, g2).*3NK, para) * 
                                cis(Form_phase(k1index, k3index, (g1,g2), spin1, 3NK)
                                + Form_phase(k2index, k4index, (-g1,-g2), spin2, 3NK))
                            end
                        end

                        normal_order_push!(list, (V,i4,i3,i2,i1))
                    end
                end



            end
        end
        return list
    end




end

# generate Hamiltonian and slove it
begin
    """
    Add mean-field terms to Hamiltonian. Only the upper triangle half.
    """
    function add_mean_field!(H,
    sorted_state_num_list::Vector{Int64}, para::LLEDPara)

        H_meanfield = para.Hmf
        k_num = para.k_num

        # H * c†_s c_s′ + h.c.
        for i in eachindex(sorted_state_num_list)
            num = sorted_state_num_list[i]
            for k in 1:k_num, kD in 1:2
    
                i_up = (k-1) + k_num*0 + 2k_num*(kD-1)
                i_dn = (k-1) + k_num*1 + 2k_num*(kD-1)
    
                # 1. same spin
                h = ComplexF64(0.0)
                if isodd(num >>> i_up)
                    h += real(H_meanfield[k,1,1,kD])
                end
                if isodd(num >>> i_dn)
                    h += real(H_meanfield[k,2,2,kD])
                end
                updateindex!(H,+,h, i,i)  
                # 2. different spin
                if isodd(num >>> i_up) && iseven(num >>> i_dn)
                    # i -> up, j -> dn
                    # num_up = num
                    num_dn = num - (1 << i_up) + (1 << i_dn)
                    j = my_searchsortedfirst(sorted_state_num_list, num_dn)
                    if j == 0
                        error("wrong block list")
                    end
                    imin, imax = (min(i_up, i_dn), max(i_up, i_dn) )
                    sign_flip = sum(ii -> (num >>> ii)%2, imin+1:imax-1; init = 0)
                    sign = isodd(sign_flip) ? -1 : 1
                    if j > i
                    updateindex!(H,+, sign * H_meanfield[k,1,2,kD], i,j)
                    else
                    updateindex!(H,+, sign * H_meanfield[k,2,1,kD], j,i)
                    end
                end
            end
        end
    end
    """
    Add one interaction term to Hamiltonian. Only the upper triangle half.
    """
    function add_interaction!(H, 
        V::ComplexF64, i4::Int64, i3::Int64, i2::Int64, i1::Int64, 
        sorted_state_num_list::Vector{Int64})

        # interaction V * c†_4 c†_3 c_2 c_1
        # i4 > i3 && i1 > i2
        # i1 >= i4
    
        a = (1 << i3) | (1 << i4)
        b = (1 << i1) | (1 << i2)
        for j in eachindex(sorted_state_num_list)
            num1 = sorted_state_num_list[j]
            isodd(num1 >>> i1) && isodd(num1 >>> i2) || continue
            num2 = num1 - b
            iseven(num2 >>> i3) && iseven(num2 >>> i4) || continue
            num3 = num2 + a
            i = my_searchsortedfirst(sorted_state_num_list, num3)
            iszero(i) && error("wrong interaction list.")
    
            sign_flip = sum(i->isodd(num2>>>i), [i2+1:i1-1; i3+1:i4-1]; init=0)
            isodd(sign_flip) && (V = -V)
    
            if i != j
                #H[i,j] += V
                #H[j,i] += conj(V)
                updateindex!(H, +, V, i,j)
                #updateindex!(H, +, conj(V), j,i)
            else
                #H[i,i] += real(V)
                updateindex!(H, +, real(V), i,i)
            end
        end
    end
    """
    Add all interaction terms to Hamiltonian. Only the upper triangle half.
    print out indicators for every N_print interaction terms
    """
    function add_all_interaction!(H, sorted_state_num_list::Vector{Int64}, 
        int_list::Vector{Tuple{ComplexF64, Int64, Int64, Int64, Int64}};
        N_print=100)
        
        for i in eachindex(int_list)
            if i%N_print == 0
                print("$i ")
            end
            add_interaction!(H, int_list[i]..., sorted_state_num_list)
        end

    end
    """
    Generate Hamiltonian of a given block
    Input: block_MBSList, interaction_list, LLED_parameter
    Parameter: print out whenever some number of interactions are added 
    """
    function LLED_Block_Hamiltonian(block_MBSList, interaction_list, para::LLEDPara;
        N_print = 100)

        size = length(block_MBSList)
        num_list = sort(getfield.(block_MBSList, :n))
        H = ExtendableSparseMatrix(ComplexF64, size, size)
        add_mean_field!(H, num_list, para)
        add_all_interaction!(H, num_list, interaction_list; N_print = N_print)
        return ExtendableSparseMatrix(Hermitian(H))
    end
    "Solve H with n eigenstates with lowest eigenenergy"
    function LLED_solve(H::ExtendableSparseMatrix, n::Int64=6)
        vec0 = rand(ComplexF64, H.cscmatrix.m)
        vals, vecs, info = eigsolve(H, vec0, n, :SR, ishermitian=true);
        return vals, vecs
    end

end


#

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



end


