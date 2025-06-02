"""
This module is designed to combine the HF and ED in two separate regions in the Brillouin zone.
The ED region is the momenta near the Dirac points and HF solves things outside.
HF gives the mean field in ED regions and ED gives the 1RDM for HF calculations.
HF + ED are calculated iteratively until they converge
"""

# module LLEDHF

using MKL
using MoireIVC.LLHF, MoireIVC.LLED







#end

