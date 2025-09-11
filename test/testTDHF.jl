using MKL
using PhysicalUnits, LinearAlgebra
using MoireIVC.LLHF
using MoireIVC.LLHF_Plot
using MoireIVC.LLTDHF
using MoireIVC: phase_color
using CairoMakie
CairoMakie.activate!()

N1 = 30; N2 = 30
@time num_para = LLHF_init_with_lambda(1.0; N1 = N1, N2 = N2, LL = 0);
LLHF_change_lambda!(num_para, 0.4);
LLHF.H0_C3_T!(num_para, 0.0);

symmetry = [LLHF.Rot3(0); LLHF.PT(0,:P)]
ρ = LLHF_solve(num_para; coherence = 0.0, 
    error_tolerance = 1E-10, max_iter_times = 300, 
    post_process_times = 300, post_procession = symmetry,
    complusive_mixing=false, complusive_mixing_rate=0.5, 
    stepwise_output = true, final_output = true
);
LLHF_plot_band(ρ; para=num_para)
GS = TDHF_groundstateanalysis(ρ, num_para);

bandplot = [ LLTDHF.band_ipl(GS, 1, x, y)
    for x in -6:0.04:66, y in -6:0.04:66
]
heatmap(bandplot)
lines([LLTDHF.band_ipl(GS, 1, x, x) for x in -6:0.04:66])


import MoireIVC.LLTDHF: TDHF_V_matrix_spin

@time Vspin = TDHF_V_matrix_spin(0.0, 0.0, num_para);

q1range = 0:1:5
omegas = zeros(ComplexF64, 4, length(q1range));
for i in eachindex(q1range)
    println(i)
    q1 = q1range[i]; q2=0
    @time M = TDHF_ZEXV(q1, q2, GS);
    vals = TDHF_solve(M)[1:8]
    perm = sortperm(vals; by = real)
    omegas[:,i] = vals[perm[5:8]]
end

begin
    magnon_spectrum = Figure()
    ax = Axis(magnon_spectrum[1,1];
        xlabel = "q1", ylabel = "ω(W0)"
    )
    for i = 1:4
        scatterlines!(ax, q1range, imag.(omegas[i,:])./num_para.system.W0)
    end
    magnon_spectrum
end
