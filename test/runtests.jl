using MoireIVC
using Test

using MoireIVC.LL_HF
num_para = LLHF_init_with_alpha(0.5; N1 = 18, N2 = 18);

@testset "MoireIVC.jl" begin
    @test typeof(num_para) == LL_HF.LLHFNumPara
end
