"""
Do Time-dependent Hartree-Fock on LLHF results.
"""
# module TDHF


using MKL, LinearAlgebra
using MoireIVC.LLHF
using MoireIVC.LLHF: LLHFNumPara, LLHFSysPara
using MoireIVC.LLHF: Form_factor, V_int
# using PhysicalUnits
using MoireIVC.Basics: ql_cross

#=
@kwdef mutable struct LLTDHFPara

end
=#
q1 = 1
q2 = 0

"""
Input the excitation momentum q=(q1, q2) and the numerical parameters
Output the interaction coefficient in form of
V[k1, k2]
"""
function interaction_coefficient(q, para::LLHFNumPara)


  


end



# end