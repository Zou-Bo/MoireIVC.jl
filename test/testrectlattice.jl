using MKL
using PhysicalUnits, LinearAlgebra
using MoireIVC.LLHF
using MoireIVC.LLHF_Plot
using MoireIVC.LLTDHF
using CairoMakie
CairoMakie.activate!()
Threads.nthreads()

AVL_sys = LLHF.LLHFSysPara(
    ϵ = 5, 
    a_Moire = 0.352nm/2.0/sind(2.1/2.0) / sqrt(2.0) * sqrt(3.0), # 0.352nm is the lattice constant of MoTe2
    D = 20nm,
    ratio12 = sqrt(3.0),
    cosθ = 0.0,
    sinθ = 1.0,
);
LLHF.define_MoTe2system().B/Tesla
LLHF.define_MoTe2system().W0
LLHF.define_MoTe2system().l
B = AVL_sys.B; B/Tesla
l = AVL_sys.l
W0 = AVL_sys.W0
Area_uc = AVL_sys.Area_uc


N1 = 16; N2 = 32;
@time AVL_para = LLHF_init_with_lambda(0.0, AVL_sys; 
    N1 = N1, N2 = N2, LL = 0
);
LLHF_change_lambda!(AVL_para, 0);
AVL_para.H0 .= 0.0;

ρVP = LLHF.VP_solution(AVL_para, 1);
ρ_AVL = LLHF_solve(AVL_para; error_tolerance = 1E-10,
    post_procession = [LLHF.Trans(1,1); LLHF.PT(0,:PT)],
    stepwise_output = false, final_output = true
);

LLHF_EnergyPerArea(ρVP; para = AVL_para) / (meV/nm^2)
LLHF_EnergyPerArea(ρ_AVL; para = AVL_para) / (meV/nm^2)

1.2666/0.1826/sqrt(27)
