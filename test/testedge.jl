include("../src/content/LL HF EDGE.jl")

using PhysicalUnits, LinearAlgebra
using MoireIVC.LLHF_Plot
using TensorCast
using CairoMakie
CairoMakie.activate!()


N1 = 3; N2 = 3
bulk_para = LLHF_init_with_alpha(1.0; N1 = N1, N2 = N2, LL = 0);
LLHF_change_lambda!(bulk_para, 0.0);
LLHF.H0_C3_T!(bulk_para, 0.0);


#
symmetry = [LLHF.Rot3(0); LLHF.PT(0,:T)]
ρ = LLHF_solve(bulk_para; coherence = 0.0, 
    error_tolerance = 1E-10, max_iter_times = 100, 
    post_process_times = 100, post_procession = symmetry,
    complusive_mixing=false, complusive_mixing_rate=0.5, 
    stepwise_output = true, final_output = true
);
#
#ρ = LLHF.VP_solution(bulk_para, 1);
LLHF_EnergyPerArea(ρ; para=bulk_para)
@reduce Xk[kx,ky] := sum(px,py) bulk_para.Fock[px,py,kx,ky,1,1];


edge_para = LLEGHF_init(bulk_para, ρ);
edge_para.λ
edge_para.Hartree .= 0.0;
sum(diag(reshape(edge_para.DMseed[:,:,:,:,2], (2N1,2N1))))
LLEGHF_EnergyPerArea(edge_para.DMseed; para=edge_para)

@reduce XE[X,XX,ky] := sum(Z,ZZ,py) edge_para.Fock[Z,ZZ,X,XX,1,1,py,ky];
XE = [XE[i,i,j] for i in 1:N1, j in 1:N2]
edge_para.system.W0
bulk_para.system.W0
edge_para.k_num

hf_interaction(edge_para.DMseed, edge_para)[:,1,:,1,:]./edge_para.system.W0

trans = [ X_kx_matrixelement(k1, X, 1, ky)
    for k1 in 0:N1-1, X in 0:N1-1, ky in 0:N2-1
]

LLEGHF_solve(edge_para; coherence = 0.0, 
    error_tolerance = 1E-10, max_iter_times = 100,
    stepwise_output = true, final_output = true
);