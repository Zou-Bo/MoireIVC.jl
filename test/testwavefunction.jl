using MKL
using PhysicalUnits, LinearAlgebra
using MoireIVC
using MoireIVC.LLHF
using Statistics
using CairoMakie
CairoMakie.activate!()

function wf_amp(wf)
    x_axis = -1.5:0.01:1.5
    y_axis = -1.1:0.01:1.1
    wavefunction = [wf(x * sys_para.a_Moire, y * sys_para.a_Moire)
        for x in x_axis, y in y_axis
    ]
    
    fig = Figure();
    ax = Axis(fig[1,1]; aspect = DataAspect())
    heatmap!(ax, x_axis, y_axis, abs.(wavefunction))
    return fig
end
function wf_ang(wf)
    x_axis = -1.5:0.01:1.5
    y_axis = -1.1:0.01:1.1
    wavefunction = [wf(x * sys_para.a_Moire, y * sys_para.a_Moire)
        for x in x_axis, y in y_axis
    ]

    fig = Figure();
    ax = Axis(fig[1,1]; aspect = DataAspect())
    heatmap!(ax, x_axis, y_axis, angle.(wavefunction);
        colormap = MoireIVC.phase_color,
    )
    return fig
end
function wf_amp_large(wf)
    x_axis = (-1.5:0.01:1.5) .* N1/2
    y_axis = (-1.1:0.01:1.1) .* N2/2
    wavefunction = [wf(x * sys_para.a_Moire, y * sys_para.a_Moire)
        for x in x_axis, y in y_axis
    ]
    
    fig = Figure();
    ax = Axis(fig[1,1]; aspect = DataAspect())
    heatmap!(ax, x_axis, y_axis, abs.(wavefunction))
    return fig
end
function wf_ang_large(wf)
    x_axis = (-1.5:0.01:1.5) .* N1/2
    y_axis = (-1.1:0.01:1.1) .* N2/2
    wavefunction = [wf(x * sys_para.a_Moire, y * sys_para.a_Moire)
        for x in x_axis, y in y_axis
    ]

    fig = Figure();
    ax = Axis(fig[1,1]; aspect = DataAspect())
    heatmap!(ax, x_axis, y_axis, angle.(wavefunction);
        colormap = MoireIVC.phase_color
    )
    return fig
end

N1 = 5; N2 = 5
num_para = LLHF_init_with_alpha(1.0; N1 = N1, N2 = N2, LL = 0);
sys_para = num_para.system;
k1 = 0; k2 = 0;
function wf_weiers(x, y; k1 = k1/N1, k2 = k2/N2)
    x /= sys_para.a_Moire
    y /= sys_para.a_Moire
    r1 = x/sqrt(0.75)
    r2 = 0.5r1 + y
    LLHF.wavefunction(r1, r2, k1, k2; sys_para=sys_para)
end
function wf_X(x, y, X; l = sys_para.l, a = sys_para.a_Moire)
    ky = X/l^2

    psiy = cis(ky*y) 
    psix = exp(-(x-X)^2 / (2.0*l^2))
    norm = sqrt(a * l) * π^0.25

    return psix*psiy/norm
end
function wf_gdct(x, y; k1=k1/N1, k2=k2/N2, symgauge = true, sys_para=sys_para, L=10)

    l = sys_para.l
    G = sys_para.G_Moire
    a = sys_para.a_Moire

    kx = (k1 + 0.5k2) * G
    ky = sqrt(0.75) * k2 * G

    psi = 0.0im
    for m in -L:L-1
        X = ky*l^2 + (m+0.5) * a * sqrt(0.75)
        psi += (1.0im)^(m*(m-1)) * cis((m+0.5)*kx*a*sqrt(0.75)) *
            wf_X(x, y, X)
    end


    psi *= cis(0.5kx*ky*l^2)
    if symgauge
        psi *= cis(-0.5x*y/l^2)
    end

    return psi
end

k1 = -0; k2 = 0;

wf_amp(wf_weiers)
wf_ang(wf_weiers)

wf_amp(wf_gdct)
wf_ang(wf_gdct)



x_range = (0:0.002:1) * sqrt(0.75) * sys_para.a_Moire
y_range = (0:0.002:1) * sys_para.a_Moire
mean(abs2.(wf_weiers.(x_range', y_range)))*sys_para.Area_uc
mean(abs2.(wf_gdct.(x_range', y_range)))*sys_para.Area_uc


function wf_r(x, y; r=3, k2=k2/N2)
    G = sys_para.G_Moire
    a = sys_para.a_Moire
    l = sys_para.l

    ky = k2 * G * sqrt(0.75)

    psi = 0.0im

    for k1 in 0:N1-1
        kx = k1/N1 * G + ky/sqrt(3)
        psi += cis(-0.5kx*ky*l^2)*cis(-(r+0.5)*kx*a*sqrt(0.75))*
            #wf_gdct(x,y; k1=k1/N1, k2=k2, symgauge = true, sys_para=sys_para, L=20)
            wf_weiers(x,y; k1=k1/N1, k2=k2)
    end

    psi /= sqrt(N1)

    return psi
end
function wf_gcr(x, y; r=3, k2=k2/N2, L=8, symgauge=true)

    l = sys_para.l
    G = sys_para.G_Moire
    a = sys_para.a_Moire

    ky = sqrt(0.75) * k2 * G

    psi = 0.0im
    for p in -L:L-1
        m = p*N1+r
        X = ky*l^2 + (m+0.5) * a * sqrt(0.75)
        psi += (1.0im)^(m*(m-1)) * cis(p*ky/sqrt(3)*N1*a*sqrt(0.75)) *
            wf_X(x, y, X)
    end

    if symgauge
        psi *= cis(-0.5x*y/l^2)
    end

    return psi

end

wf_amp_large(wf_r)
wf_ang_large(wf_r)


wf_amp_large(wf_gcr)
wf_ang_large(wf_gcr)