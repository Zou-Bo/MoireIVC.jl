using MKL, LinearAlgebra
using PhysicalUnits
using MoireIVC
using CairoMakie
CairoMakie.activate!()
lines(1:10, 5:14)

using MoireIVC.LLHF, MoireIVC.LLED
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
function DiracCone(E_1=E_1, k_index=k_index)
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
Hmf=DiracCone()





para1 = LLED_init(k_index, Hmf, system, 30, 1);
blocks1, block_k1_1, block_k2_1, bn0_1 = 
LLED_block_bysmallmomentum(LLED_mbslist_twoDiracCone(para1, 7, 7);
    k1range=(-2,2), k2range=(-2,2)
);
blocks1[bn0_1]
int_list1= LLED_interactionlist(para1)



##########################################
# plot eigen spectrum of different momenta


N_eigen = 8
eigenenergy = Matrix{Float64}(undef, N_eigen, length(blocks1));
for bn in eachindex(blocks1)
    println(bn, "\t", length(blocks1[bn]))
    @time H = LLED_Block_Hamiltonian(blocks1[bn], int_list1, para1)
    @time vals, vecs = LLED_solve(H, N_eigen)
    if length(vals) >= N_eigen
        eigenenergy[:,bn] = real.(vals[1:N_eigen])
    else
        eigenenergy[1:length(vals),bn] = real.(vals)
        eigenenergy[1+length(vals):N_eigen,bn] = NaN
    end
end

begin
    eigEs = Figure(size = (800,600));
    ax = Axis(eigEs[1,1],
        xlabel = "k1+5k2",
    )
    for bn in eachindex(blocks1)
        k1 = block_k1_1[bn]; k2 = block_k2_1[bn]
        xticknumber = k1 + 5k2
        energy_shift = E_1 * sqrt(k1^2 + k2^2 + k1*k2)
        for j in 1:N_eigen
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
    eigEs
end
begin
    lowest_E = Figure(size = (800,600));
    ax_le = Axis(lowest_E[1,1])
    Emin = Base.vec(eigenenergy[1,:])
    block_kx = block_k1_1 .* system.G1[1] + block_k2_1 .* system.G2[1]
    block_ky = block_k1_1 .* system.G1[2] + block_k2_1 .* system.G2[2]
    scatter!(ax_le, block_kx, block_ky;
        color = Emin, markersize = 20
    )
    lowest_E
end










#################################################
# calculate and analysis the k=0 ground states

@time H_k0 = LLED_Block_Hamiltonian(blocks1[bn0_1], int_list1, para1)
ishermitian(H_k0)
@time vals_k0, vecs_k0 = LLED_solve(H_k0, 10);
vals_k0
dot(vecs_k0[1], vecs_k0[2])


onebodyRDM = LLED.one_body_reduced_density_matrix(vecs_k0[1], blocks1[bn0_1])
tr(onebodyRDM)/14
maximum(abs.(onebodyRDM))
heatmap(abs.(onebodyRDM), colorrange = (0,1))


using Combinatorics
iA1 = [0;7];
nA1 = (i_set -> sum(1 .<< i_set; init = 0)).(collect(powerset(iA1)));
iA2 = [14;21];
nA2 = (i_set -> sum(1 .<< i_set; init = 0)).(collect(powerset(iA2)));
iA3 = [0;7;14;21];
nA3 = (i_set -> sum(1 .<< i_set; init = 0)).(collect(powerset(iA3)));
iA4 = collect(0:13);
nA4 = getfield.(LLED.mbslist_oneDiracCone(para1, 7), :n);

vec0 = vecs_k0[1];
RDM_A = LLED.reduced_density_matrix(vec0, blocks1[bn0_1], nA1, iA1);
LLED.entanglement_entropy(RDM_A)
RDM_A = LLED.reduced_density_matrix(vec0, blocks1[bn0_1], nA2, iA2);
LLED.entanglement_entropy(RDM_A)
RDM_A = LLED.reduced_density_matrix(vec0, blocks1[bn0_1], nA3, iA3);
LLED.entanglement_entropy(RDM_A)
RDM_A = LLED.reduced_density_matrix(vec0, blocks1[bn0_1], nA4, iA4);
LLED.entanglement_entropy(RDM_A)

1

#=
sum(i -> (abs2(psi1[i])>1e-6), eachindex(psi1); init = 0)
lines(diff(log.(sort(abs2.(psi1))))[end-10000:end])

lines(abs2.(vecsK0[1]))
lines(abs2.(vecsK0[2]))
lines(abs2.(vecsK0[3]))
lines(abs2.(vecsK0[4]))
lines(abs2.(vecsK0[5]))
lines(abs2.(vecsK0[6]))

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
=#



#####################################################
# only the two Dirac points
para2 = LLED_init(k_index[:,[1]], Hmf[[1],:,:,:], system, 30, 1);


blocks2 = 
LLED_block_bysmallmomentum(LLED_mbslist_twoDiracCone(para2, 1, 1);
    momentum_restriction = false
)[1]
int_list2 = LLED_interactionlist(para2)

H2_k0 = LLED_Block_Hamiltonian(blocks2[1], int_list2, para2)
vals2_k0, vecs2_k0 = eigen(Matrix(H2_k0))


vec0 = vecs2_k0[:,3];
iA = [0;1]; nA = [1;2];
RDM_A = LLED.reduced_density_matrix(vec0, blocks2[1], nA, iA);
LLED.entanglement_entropy(RDM_A)

