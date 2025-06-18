using MoireIVC
using Test


using MoireIVC.LLHF
N1 = 1
N2 = 1
num_para = LLHF_init_with_lambda(0.3; N1 = N1, N2 = N2);
LLHF_change_lambda!(num_para, 0.2);
symmetry = [LLHF.Rot3(0); LLHF.PT(0,:PT)]
LLHF.H0_C3_T!(num_para, 0.1);
ρ = LLHF_solve(num_para; coherence = 0.0, 
    error_tolerance = 1E-10, max_iter_times = 100, 
    post_process_times = 100, post_procession = symmetry,
    complusive_mixing=false, complusive_mixing_rate=0.5, 
    stepwise_output = true, final_output = true
);


@testset "MoireIVC.jl" begin
    @test typeof(num_para) == LLHF.LLHFNumPara
    @test abs(ρ[1,1,1,1] - 0.5) < 1e-10
end
