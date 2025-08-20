using MKL, LinearAlgebra
using PhysicalUnits
using MoireIVC.LLHF
using MoireIVC.LLED
using CairoMakie
CairoMakie.activate!()

system = LLHF.define_MoTe2system();
k_index = [
     0;  0;;
    +1;  0;;
     0; +1;;
    -1; +1;;
    -1;  0;;
     0; -1;;
    +1; -1;;

    +1; +1;;
    +2;  0;;
    +2; -1;;
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
        #error("k_num should be 0 or 7")
        return Hmf
    end
end
Hmf=DiracCone()


para1 = LLED_init(k_index[:,:], Hmf[:,:,:,:], system, 30, 1);
para1.k_num
LLED.mbslist_oneDiracCone(para1, para1.k_num)


blocks1, block_k1_1, block_k2_1, bn0_1 = 
LLED_block_bysmallmomentum(LLED_mbslist_twoDiracCone(para1, 10, 10);
    k1range=(-2,2), k2range=(-2,2)
);
blocks1[bn0_1];
int_list1= LLED_interactionlist(para1)
int_list_array = Matrix{Union{ComplexF64, Int64}}(undef, 121, 5)
for i in eachindex(int_list1)
    int_list_array[i, :] .= int_list1[i][1:5]
end
findfirst(x->x[2:5]==(3,0,4,11), int_list1)