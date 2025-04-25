using MoireIVC
using Test

using MoireIVC: LLHF_solve, LLHF_init_with_alpha
LLHF_solve(LLHF_init_with_alpha(0.5))

@testset "MoireIVC.jl" begin
    @test true
end
