module MoireIVC
using MKL
using CairoMakie

public phase_color
phase_color = range(Makie.Colors.HSV(0,1,1), stop=Makie.Colors.HSV(360,1,1), length=90);


include("contents/basicthings.jl")
include("methods/HartreeFock.jl")


include("contents/LL HF.jl")
#using .LLHF
include("plots/LL HF plot.jl")
#using .LLHF_Plot
include("contents/LL TDHF.jl")
#using .LLTDHF

include("contents/LL ED.jl")
# include("contents/LL ED old.jl")
#using .LLED





end
