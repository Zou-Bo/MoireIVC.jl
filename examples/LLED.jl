

using MoireIVC.LLHF
system = LLHF.define_system();
k_index = [
     0;  0;;
    +1;  0;;
     0; +1;;
    -1; +1;;
    -1;  0;;
     0; -1;;
    +1; -1;;
]
E_1 = 5.37meV
# Hmf[k,s,s',kD] c†_{k,s,kD} c_{k,s',kD};
function symmetrize_meanfield!(E_1=E_1, k_index=k_index)
    k_num = size(k_index,2)

    Hmf = zeros(ComplexF64, k_num, 2, 2, 2)
    if k_num == 1
        return Hmf
    elseif k_num == 7

        h1diag = 0.0
        h1offd = E_1
        for i in 1:6
            phase = ((i-1)/3-1/6)*pi
            Hmf[1+i,:,:,1] .=[h1diag;  h1offd*cis(phase);;  h1offd*cis(-phase); h1diag] 
            Hmf[1+i,:,:,2] .=[h1diag; -h1offd*cis(phase);; -h1offd*cis(-phase); h1diag] 
        end

        return Hmf
    else
        error("k_num should be 0 or 7")
    end
end
Hmf=symmetrize_meanfield!()
para = LLED_init(k_index[:,[1,4,7]], Hmf[[1,4,7],:,:,:], system, 30, 1);

blocks, block_k1, block_k2, bn0 = 
Block_smallmomentum(MBSList_twoDiracCone(para, 3, 3));
blocks[bn0]
InteractionList = interaction_list(para)





##########################################


eigenenergy = Matrix{Float64}(undef, 10, length(blocks));
for bn in eachindex(blocks)
    println(bn, "\t", length(blocks[bn]))
    @time H = Block_Hamiltonian(blocks[bn], InteractionList, para)
    vec0 = rand(ComplexF64, length(blocks[bn]))
    @time vals, vecs, info = eigsolve(H, vec0, 10, :SR, ishermitian=true);
    if length(vals) >= 10
        eigenenergy[:,bn] = real.(vals[1:10])
    else
        eigenenergy[1:length(vals),bn] = real.(vals)
        eigenenergy[1+length(vals):10,bn] = NaN
    end
end

using CairoMakie
CairoMakie.activate!()
begin
    eigEs = Figure(size = (800,600));
    ax = Axis(eigEs[1,1],
        xlabel = "k1+5k2",
    )
    for bn in eachindex(blocks)
        k1 = block_k1[bn]; k2 = block_k2[bn]
        xticknumber = k1 + 5k2
        energy_shift = E_1 * sqrt(k1^2 + k2^2 + k1*k2)
        for j in 1:10
            if !isnan(eigenenergy[j, bn])
                scatter!(ax, xticknumber, eigenenergy[j, bn];
                    color=:black, marker=:hline
                )
                if xticknumber != 0
                    scatter!(ax, xticknumber, eigenenergy[j, bn]-energy_shift;
                        color=:blue, marker=:hline
                    )
                end
            end
        end
    end
    #ylims!(-555, -550)
    eigEs
end
begin
    lowest_E = Figure(size = (800,600));
    ax_le = Axis(lowest_E[1,1])
    Emin = vec(eigenenergy[1,:])
    block_kx = block_k1 .* system.G1[1] + block_k2 .* system.G2[1]
    block_ky = block_k1 .* system.G1[2] + block_k2 .* system.G2[2]
    scatter!(ax_le, block_kx, block_ky;
        color = Emin, markersize = 20
    )
    lowest_E
end










#################################################

@time HK0 = Block_Hamiltonian(blocks[bn0], InteractionList, para)
ishermitian(HK0)
@time valsK0, vecsK0, info = eigsolve(HK0, rand(ComplexF64, HK0.cscmatrix.n), 10, :SR, ishermitian=true);
valsK0


psi1 = vecsK0[1];
psi2 = vecsK0[2];
psi3 = vecsK0[3];
psi4 = vecsK0[4];
psi5 = vecsK0[5];
psi6 = vecsK0[6];

sum(i -> (abs2(psi1[i])>1e-6), eachindex(psi1); init = 0)
lines(diff(log.(sort(abs2.(psi1))))[end-10000:end])

lines(abs2.(psi1))
lines(abs2.(psi2))
lines(abs2.(psi3))
lines(abs2.(psi4))
lines(abs2.(psi5))
lines(abs2.(psi6))

blocks[bn0][7]

function find_index(spin1, spin2, mbs_list = blocks[bn0])
    index = Int64[]
    i1 = 7(spin1-1)
    i2 = 7(spin2-1)+14
    for i in eachindex(mbs_list)
        if any(==(i1), mbs_list[i].i) && any(==(i2), mbs_list[i].i)
            push!(index, i)
        end
    end
    return index
end
(maximum(abs2.(psi1)[find_index(1,1)]) * 2^12 +
 maximum(abs2.(psi1)[find_index(1,2)]) * 2^12 +
 maximum(abs2.(psi1)[find_index(2,1)]) * 2^12 +
 maximum(abs2.(psi1)[find_index(2,2)]) * 2^12 ) 
1



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

onebodyRDM = one_body_reduced_density_matrix(vecsK0[4], blocks[bn0])
tr(onebodyRDM)/6
heatmap(abs.(onebodyRDM), colorrange = (0,1))
maximum(abs.(onebodyRDM))
dot(vecsK0[1], vecsK0[2])



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
    println(NA)
    RDM_threads = zeros(ComplexF64, NA, NA, Threads.nthreads())
    #Threads.@threads 
    for nchunk in 1:length(Bchunks_lastindices)-1
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

iA = [6;9]
nA = (i_set -> sum(1 .<< i_set; init = 0)).(collect(powerset(iA)));
nA = getfield.(MBSList_oneDiracCone(para, 3), :n)
RDM_A = reduced_density_matrix(vecsK0[1], blocks[bn0], nA, iA);
tr(RDM_A)
ishermitian(RDM_A)
(eigvals(RDM_A))
sum(eigvals(RDM_A)) do x
    if abs(x) < 1E-5
        return 0.0
    end
    -x * log2(x)
end


#####################################################
para0 = LLED_init(k_index[:,[1]], Hmf[[1],:,:,:], system, 30, 1);

try_mbslist = MBSList_twoDiracCone(para0, 1, 1)
blocks00 = 
Block_smallmomentum(MBSList_twoDiracCone(para0, 1, 1))[1][1]
InteractionList0 = interaction_list(para0)

H00 = Block_Hamiltonian(blocks00, InteractionList0, para0)
vals2, vecs2 = eigen(Matrix(H00))




num_list = sort(getfield.(blocks00, :n))
H = ExtendableSparseMatrix(ComplexF64, 4, 4)
add_mean_field!(H, num_list, para0)
#add_all_interaction!(H, num_list, interaction_list; N_print = N_print)
add_interaction!(H, InteractionList0[3]..., num_list)
H

