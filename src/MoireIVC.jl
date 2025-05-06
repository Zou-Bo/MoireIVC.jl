module MoireIVC
using MKL
using CairoMakie

public phase_color
phase_color = range(Makie.Colors.HSV(0,1,1), stop=Makie.Colors.HSV(360,1,1), length=90);


include("content/basicthings.jl")


include("content/LL HF.jl")
#using .LLHF
include("plot/LL HF plot.jl")
#using .LLHF_Plot
include("content/LL TDHF.jl")
#using .LLTDHF

include("content/LL ED.jl")
#using .LLED





end
