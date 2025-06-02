"""
This modele gives general methods for momentum resolved ED calculations
Sectors of other quantum Numbers should be handled outside,
this module only set sectors of total (crystal) momentum.
"""
#module MomentunResolvedExactDiagonalization

using MKL
using Combinatorics
using KrylovKit

@kwdef mutable struct EDPara
    k_list::Matrix{Int64}

    Nk::Int64 = size(k_list,2)
    Nc::Int64 = 1  # number of components

    # conserved momentum mod G, where G=0 means no mod
    Gk::Tuple{Int64, Int64} = (0, 0)

    # onebody / twobody hamiltonian
    H1::Matrix{ComplexF64} = zeros(ComplexF64, Nk*Nc, Nk*Nc)
    H2::Vector{Tuple{ComplexF64,Int64,Int64,Int64,Int64}} = []

end


"""
The struct of many-body state consists of 
    n: the integer consist of bit array
    i: sorted vector of occupies bit numbers
    k1, k2: total momentum
n = sum(i -> 1<<i, i_list; init=0)
i = i_k + Nk * ic
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
Return the total momentum K1 and K2 of a many body state.
"""
function TotalMomentum(para::EDPara, i_list::Vector{Int})
    k1 = 0; k2 = 0; Gk = para.Gk
    for i in i_list
        momentum = para.k_list[:, i%para.Nk + 1]
        k1 += momentum[1]
        k2 += momentum[2]
    end
    if !iszero(Gk[1])
        k1 = mod(k1, Gk[1])
    end
    if !iszero(Gk[2])
        k2 = mod(k2, Gk[2])
    end
    return k1, k2
end








"""
construct the mbs with N electrons in one component.
"""
function mbslist_onecomponent(para::EDPara, N_in_one::Int64)
    Nk = para.Nk
    N_in_one > Nk && error("the maximum number of electrons in one component is $Nk")
    N_in_one < 0 && error("the number of electrons cannot be negative")
    sort([MBS(sum(1 .<< combi), combi, TotalMomentum(para,combi)..., Nk)
        for combi in collect(combinations(0:Nk-1, N_in_one))
    ])
end
"""
return a list of MBS with electron number [N1, N2, ...] in each components.
"""
function ED_mbslist(para::LLEDPara, N_each_component::Vector{Int64})
    length(N_each_component) > para.Nc && error("max number of components is $(para.Nc)")
    list = mbslist_oneDiracCone(para, N_each_component[begin])
    for i in eachindex(N_each_component)[2:end]
        list = kron(mbslist_oneDiracCone(para, N_each_component[i]), list)
    end
    return list
end
"""
Divide a given mbs list into blocks with same momentum
Input: MBS list
    Output:
    1. list of block MBS list,
    2. list of k1 of blocks,
    3. list of k2 if blocks
"""
function ED_block_division(mbs_list::Vector{MBS};
    momentum_restriction = false, k1range=(-2,2), k2range=(-2,2),
    momentum_list::Vector{Tuple(Int64, Int64)} = [], )

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

    if isempty(momentum_list)
        for K1 in k1min:k1max, K2 in k2min:k2max
            mask = findall(mbs -> mbs.k1==K1 && mbs.k2==K2, mbs_list)
            if length(mask) != 0
                push!(blocks, mbs_list[mask])
                push!(block_k1, K1)
                push!(block_k2, K2)
            end
        end
    else
        for K1 in k1min:k1max, K2 in k2min:k2max
            if  (K1, K2) ∈ momentum_list
                mask = findall(mbs -> mbs.k1==K1 && mbs.k2==K2, mbs_list)
                if length(mask) != 0
                    push!(blocks, mbs_list[mask])
                    push!(block_k1, K1)
                    push!(block_k2, K2)
                end
            end
        end
    end
    return blocks, block_k1, block_k2
end
















function normal_order!()
end

function normalize_interaction_list()
end

function add_one_body!(H, H0; klist)
end

function add_interaction!(H, Hint; klist)
end

function EDHamiltonian(para::EDPara)
end

function EDsolve(H)








end