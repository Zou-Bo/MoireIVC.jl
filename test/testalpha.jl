using MKL
using PhysicalUnits, LinearAlgebra
using MoireIVC.LLHF
using MoireIVC.LLHF_Plot
using MoireIVC.LLTDHF
using CairoMakie
CairoMakie.activate!()


N1 = 18; N2 = 18
num_para = LLHF_init_with_alpha(1.0; N1 = N1, N2 = N2, LL = 0);
LLHF_change_lambda!(num_para, 0.5);
LLHF.H0_C3_T!(num_para, 0.0);
ρVP = LLHF.VP_solution(num_para,1);
LLHF_plot_band_3D(ρVP; para=num_para)


symmetry = [LLHF.Rot3(0); LLHF.PT(0,:T)]
ρ = LLHF_solve(num_para; coherence = 0.0, 
    error_tolerance = 1E-10, max_iter_times = 100, 
    post_process_times = 100, post_procession = symmetry,
    complusive_mixing=false, complusive_mixing_rate=0.5, 
    stepwise_output = true, final_output = true
);
LLHF_EnergyPerArea(ρ; para=num_para)
LLHF_EnergyPerArea(ρVP, para = num_para)

LLHF.trace(ρ, 0.5LLHF.hf_interaction(ρ, num_para, :HA)) / num_para.k_num / num_para.system.Area_uc
LLHF.trace(ρ, 0.5LLHF.hf_interaction(ρ, num_para, :HE)) / num_para.k_num / num_para.system.Area_uc
LLHF.trace(ρ, 0.5LLHF.hf_interaction(ρ, num_para, :XA)) / num_para.k_num / num_para.system.Area_uc
LLHF.trace(ρ, 0.5LLHF.hf_interaction(ρ, num_para, :XE)) / num_para.k_num / num_para.system.Area_uc

LLHF.trace(ρVP, 0.5LLHF.hf_interaction(ρVP, num_para, :HA)) / num_para.k_num / num_para.system.Area_uc
LLHF.trace(ρVP, 0.5LLHF.hf_interaction(ρVP, num_para, :HE)) / num_para.k_num / num_para.system.Area_uc
LLHF.trace(ρVP, 0.5LLHF.hf_interaction(ρVP, num_para, :XA)) / num_para.k_num / num_para.system.Area_uc
LLHF.trace(ρVP, 0.5LLHF.hf_interaction(ρVP, num_para, :XE)) / num_para.k_num / num_para.system.Area_uc


LLHF_plot_band_3D(ρ; para=num_para)
LLHF_plot_band(ρ; para=num_para)
LLHF_plot_phase(ρ; para=num_para)
LLHF_plot_Sz(ρ; para=num_para)
LLHF_plot_Berrycurvature(ρ; para=num_para)