using MoireIVC
using Test


using MoireIVC.LLHF
N1 = 3
N2 = 3
num_para = LLHF_init_with_lambda(1.0; N1 = N1, N2 = N2);




@testset "MoireIVC.jl" begin
    @test typeof(num_para) == LL_HF.LLHFNumPara
end
