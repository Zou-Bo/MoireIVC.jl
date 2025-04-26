module LLHF_Plot

using MKL
using LinearAlgebra
using PhysicalUnits
using CairoMakie, GLMakie, Printf
CairoMakie.activate!()
using MoireIVC.LLHF
using MoireIVC.LLHF: LLHFNumPara, LLHFSysPara
using MoireIVC.LLHF: polar_azimuthal_angles, berry_curvature, realspace_pauli
using MoireIVC.Basics: ql_cross
import MoireIVC.phase_color

export LLHF_plot_band_3D, LLHF_plot_band, LLHF_plot_realspace
export LLHF_plot_phase, LLHF_plot_Sz, LLHF_plot_Berrycurvature





function LLHF_plot_band_3D(rho; para::LLHFNumPara, unit=:W0)

    N1 = para.N1; N2 = para.N2
    G1 = para.system.G1; G2 = para.system.G2

    if unit == :W0
        u = para.system.W0
    else
        u = meV
    end

    bands = LLHF.band(rho; para = para)
    plot_x = Matrix{Float64}(undef, N1, N2);
    plot_y = Matrix{Float64}(undef, N1, N2);
    for k1 in 0:N1-1, k2 in 0:N2-1
        plot_x[1+k1, 1+k2], plot_y[1+k1, 1+k2] = k1/N1*G1 + k2/N2*G2
    end
    GLMakie.activate!()
    fig, ax, sf1 = surface(plot_x, plot_y, bands[:,:,1]./u; 
        axis=(;type=Axis3, aspect=(sqrt(3),1,1), 
            zlabel="E (" * String(unit) * ")"
        ),
    )
    sf2 = surface!(ax, plot_x, plot_y, bands[:,:,2]./u;
        colormap = phase_color,
        colorrange = (-π, π),
        color = [ angle(rho[1+k1, 1+k2,2,1])
            for k1 in 0:N1-1, k2 in 0:N2-1
        ],
    )
    #display(current_figure())
    display(fig)
    CairoMakie.activate!()
end
function LLHF_plot_band(rho; para::LLHFNumPara, unit=:W0)

    N1 = para.N1; N2 = para.N2
    N1 != N2 && error("LLHF_plot_band only works with N1=N2")

    if unit == :W0
        u = para.system.W0
    else
        u = meV
    end

    bands = LLHF.band(rho; para=para)
    b1 = diag(bands[:,:,1])./u
    b1 = [b1; bands[1:N1÷2+1,1,1]./u]
    b2 = diag(bands[:,:,2])./u
    b2 = [b2; bands[1:N1÷2+1,1,2]./u]


    fig = Figure(size=(500,400))
    ax = Axis(fig[1,1],
        xticks=([0,N1/3,2N1/3,N1,N1+N1÷2], ["γ", "κ", "κ'", "γ", "m"]), 
        xticksvisible = false,
        xgridcolor = :black,
        ylabel="E (" * String(unit) * ")",
        limits=((0,N1+N1÷2), nothing)
    )
    lines!(ax, 0:N1+N1÷2, b1, color=:blue)
    lines!(ax, 0:N1+N1÷2, b2, color=:blue)

    display(fig)

end




function LLHF_plot_realspace(ρ; para::LLHFNumPara,
    N = 36, xlimits = (-1.2, 1.2), ylimits = (-1.2, 1.2), 
    arrowscale = 1, arrowsize = 4, 
    colormap = :berlin, colorrange = nothing,
    colorbar = true, text = true, textsize = 20,
    text_position=([0.0; 1.1], [0.0; 1.0], [0.0, 0.9]),
    )

    a = para.system.a_Moire
    a1 = para.system.a1
    a2 = para.system.a2

    fig = Figure(size = (780,700));

    N = (N ÷ 3) * 3
    dx = 1. / N
    dy = sqrt(3) * dx
    
    x1_range = [reverse(collect(-dx:-dx:xlimits[1])); collect(0.0:dx:xlimits[2])]
    x2_range = [reverse(collect(-dx/2.0:-dx:xlimits[1])); collect(dx/2.0:dx:xlimits[2])]
    y1_range = [reverse(collect(-dy:-dy:ylimits[1])); collect(0.0:dy:ylimits[2])]
    y2_range = [reverse(collect(-dy/2.0:-dy:ylimits[1])); collect(dy/2.0:dy:ylimits[2])]
    scale = arrowscale * dx

    plot_x = Vector{Float64}(undef,length(x1_range)*length(y1_range)+length(x2_range)*length(y2_range))
    plot_y = similar(plot_x)
    plot_Sx= similar(plot_x)
    plot_Sy= similar(plot_x)
    plot_Sz= similar(plot_x)
    index::Int64 = 1
    for (i, j) in Iterators.product(eachindex(x1_range), eachindex(y1_range))
        x = x1_range[i]
        y = y1_range[j]
        plot_x[index] = x
        plot_y[index] = y
        plot_Sx[index], plot_Sy[index], plot_Sz[index] = realspace_pauli(x*a, y*a, ρ, para)[[1,2,3]]
        index += 1
    end
    for (i, j) in Iterators.product(eachindex(x2_range), eachindex(y2_range))
        x = x2_range[i]
        y = y2_range[j]
        plot_x[index] = x
        plot_y[index] = y
        plot_Sx[index], plot_Sy[index], plot_Sz[index] = realspace_pauli(x*a, y*a, ρ, para)[[1,2,3]]
        index += 1
    end

    ax = Axis(fig[1,1], aspect = 1.0, 
        xticksvisible = true, yticksvisible = true,
        xgridvisible = true, ygridvisible = true,
        xticklabelsvisible = true, yticklabelsvisible = true, 
    )
    if isnothing(colorrange)
        colorrange = (minimum(plot_Sz)-0.01, maximum(plot_Sz)+0.01)
    end
    ar = arrows!(ax, plot_x, plot_y, plot_Sx, plot_Sy, 
        color = plot_Sz, colorrange = colorrange, colormap = colormap,
        arrowsize = arrowsize, lengthscale = scale,
    )
    xlims!(ax, xlimits...)
    ylims!(ax, ylimits...)
    colorbar && Colorbar(fig[1,2], ar)

    primitive_cell = zeros(Float64, 2, 5)
    primitive_cell[:,1] = -0.5a1 - 0.5a2
    primitive_cell[:,5] = -0.5a1 - 0.5a2
    primitive_cell[:,2] =  0.5a1 - 0.5a2
    primitive_cell[:,3] =  0.5a1 + 0.5a2
    primitive_cell[:,4] = -0.5a1 + 0.5a2
    primitive_cell ./= a
    lines!(ax, primitive_cell[1,:], primitive_cell[2,:],
        color = :red
    )



    if text
        text!(ax, text_position[1]...; fontsize = textsize,
            text = (@sprintf "Sx: %.4f ~ %.4f" minimum(plot_Sx) maximum(plot_Sx))
        )
        text!(ax, text_position[2]...; fontsize = textsize,
            text = (@sprintf "Sy: %.4f ~ %.4f" minimum(plot_Sy) maximum(plot_Sy))
        )
        text!(ax, text_position[3]...; fontsize = textsize,
            text = (@sprintf "Sz: %.4f ~ %.4f" minimum(plot_Sz) maximum(plot_Sz))
        )
    end
    fig
end




function hexgon_heatmap!(ax, range1, range2, colormatrix;
    period1, period2, # G1, G2 
    colorrange=(minimum(colormatrix)-0.01, maximum(colormatrix)+0.01), colormap=:viridis, 
    hexgonsize = abs(range1[2]-range1[1])/sqrt(3), 
    hexgonrotate = angle(period1[1]+period1[2]*im), 
    hexgon_expension = 0.02, others...
    )

    hexgon_edge = hexgonsize * sqrt(abs2(period1[1])+abs2(period1[2]))
    hexgon_edge *= 1.0 + hexgon_expension
    hexagon = Makie.Polygon([Point2f(cos(θ), sin(θ)) 
        for θ in range(π/6, 13π/6, length = 7) .+ hexgonrotate]
    )
    points = vec([Point2f((i*period1 + j*period2)...) for i in range1, j in range2])
    scatter!(ax, points; color = vec(colormatrix),
        marker = hexagon, markersize = hexgon_edge, markerspace = :data,
        colormap = colormap, colorrange = colorrange, others...
    )
end
function periodic_hexgon_BZ(matrix; G1, G2, mid_mesh=false, 
    colorrange=(minimum(matrix)-0.01, maximum(matrix)+0.01),
    colormap=:viridis, ticks=Makie.automatic, others...
    )

    N1, N2 = size(matrix)

    k1range = floor(Int64,-1.2N1):floor(Int64,1.2N1)
    k2range = floor(Int64,-0.9N2):floor(Int64,0.9N2)

    fig = Figure(size=(600,500));
    data = [ matrix[1+k1-floor(Int64,k1//N1)*N1, 1+k2-floor(Int64,k2//N2)*N2]
        for k1 in k1range, k2 in k2range
    ]

    if mid_mesh
        k1range = k1range .+ 0.5
        k2range = k2range .+ 0.5
    end

    ax = Axis(fig[1,1], aspect = DataAspect())
    hm = hexgon_heatmap!(ax, k1range./N1, k2range./N2, data;
        period1 = G1, period2 = G2,
        colormap = colormap, colorrange = colorrange, others...
    )
    Colorbar(fig[1,2], hm, ticks=ticks)

    BZpoints = [Point2f([cospi(1/3) -sinpi(1/3); sinpi(1/3) cospi(1/3)]^n * (G1+G2)/3.0)
        for n = 0:6
    ]
    lines!(ax, BZpoints; color=:white)

    L = 0.7norm(G1)
    ylims!(-L,L)
    xlims!(-L,L)

    display(fig)



end


function LLHF_plot_phase(ρ; para::LLHFNumPara, others...)

    N1 = para.N1; N2 = para.N2
    G1 = para.system.G1; G2 = para.system.G2
    
    k1range = floor(Int64,-1.2N1):floor(Int64,1.2N1)
    k2range = floor(Int64,-0.9N2):floor(Int64,0.9N2)

    fig = Figure(size=(600,500));
    al = [ angle(ρ[1+k1-floor(Int64,k1//N1)*N1, 1+k2-floor(Int64,k2//N2)*N2,2,1])
        for k1 in k1range, k2 in k2range
    ]
    for I in CartesianIndices(al)
        k1 = k1range[I[1]]; k2 = k2range[I[2]];
        al[I] += ql_cross(floor(k1//N1), floor(k2//N2), k1/N1, k2/N2)
        while al[I] <= -π
            al[I] += 2π
        end
        while al[I] > π
            al[I] -= 2π
        end
    end
    ax = Axis(fig[1,1], aspect = DataAspect())
    hexgon_heatmap!(ax, k1range./N1, k2range./N2, al;
        period1 = G1, period2 = G2,
        colormap = phase_color, colorrange = (-pi,pi), others...
    )
    Colorbar(fig[1,2], colormap = phase_color, colorrange = (-pi,pi),
        ticks = (-pi:0.5pi:pi, ["-π","-π/2","0","π/2","π"])
    )

    BZpoints = [Point2f([cospi(1/3) -sinpi(1/3); sinpi(1/3) cospi(1/3)]^n * (G1+G2)/3.0)
        for n = 0:6
    ]
    lines!(ax, BZpoints; color=:white)

    L = 0.7norm(G1)
    ylims!(-L,L)
    xlims!(-L,L)

    display(fig)
end
function LLHF_plot_Sz(ρ; para::LLHFNumPara, others...)

    theta, phi = polar_azimuthal_angles(ρ, para)
    periodic_hexgon_BZ(cos.(theta); G1 = para.system.G1, G2 = para.system.G2,
        colormap = :RdBu, colorrange = (0,1), others...
    )

end
function LLHF_plot_Berrycurvature(ρ; para::LLHFNumPara, others...)

    BerryCurvature = berry_curvature(ρ, para)
    @show ChernNumber = sum(BerryCurvature)
    periodic_hexgon_BZ(BerryCurvature*2π; G1 = para.system.G1, G2 = para.system.G2,
        colormap = :RdBu, mid_mesh = true, others...
    )


end








# depreciated

# plotting density matrix ρ[τ = 2, τ′ = 1] = c†_1 c_2 
function LLHF_plot_DM_phase(ρ; para::LLHFNumPara,
    edge1 = round(Int64, para.N1/2), edge2 = round(Int64, para.N2/2),)

    N1 = para.N1; N2 = para.N2
    
    fig = Figure(size=(1000,900));
    al = [ angle(ρ[1+(k1+N1)%N1, 1+(k2+N2)%N2,2,1])
    for k1 in -edge1:N1+edge1, k2 in -edge2:N2+edge2
    ]
    for I in CartesianIndices(al)
        k1 = I[1] - edge1-1; k2 = I[2] - edge2-1;
        if k1 < 0
            al[I] += ql_cross(k1/N1, k2/N2, 1,0)
        elseif k1 >= N1
            al[I] += ql_cross(k1/N1, k2/N2,-1,0)
        end
        if k2 < 0
            al[I] += ql_cross(k1/N1, k2/N2, 0,1)
        elseif k2 >= N2
            al[I] += ql_cross(k1/N1, k2/N2, 0,-1)
        end
        if al[I]<-π
            al[I] += 2π
        elseif al[I]>π
            al[I] -= 2π
        end
    end
    (ax, hm) = heatmap(fig[1,1], -edge1:N1+edge1, -edge2:N2+edge2, al,
        colormap = phase_color, colorrange = (-pi,pi),
    )
    Colorbar(fig[1,2], hm, ticks = (-pi:0.5pi:pi, ["-π","-π/2","0","π/2","π"]))

    vlines!(ax, 0, color=:white)
    vlines!(ax, N1, color=:white)
    hlines!(ax, 0, color=:white)
    hlines!(ax, N2, color=:white)

    scatter!(ax, 0.25N1, 0.75N2, marker=:diamond, color=:white)
    scatter!(ax, 0.75N1, 0.25N2, marker=:diamond, color=:white)

    scatter!(ax, N1/3, N2/3, marker=:diamond, color=:white)
    scatter!(ax, 2N1/3, 2N2/3, marker=:diamond, color=:white)

    display(fig)
end


#=
    # plotting fock coefficients X[τn′ = 1, τn = 2]
    function plot_Fock_phase(k1, k2; N1 = N1, N2 = N2, moduleis = IVC,
        edge1 = round(Int64, N1/2), edge2 = round(Int64, N2/2),
        phasecolor = range(Makie.Colors.HSV(0,1,1), stop=Makie.Colors.HSV(360,1,1), length=90))

        println("Fock[p1=$k1,p2=$k2,k1=$k1,k2=$k2,1,1,1,1,1,2] = ", moduleis.Fock[1+k1,1+k2,1+k1,1+k2,1,1,1,1,1,2])
        fig = Figure(size=(1000,900));
        al = [ angle(moduleis.Fock[1+(p1+N1)%N1, 1+(p2+N2)%N2,1+k1,1+k2,1,1,1,1,1,2])
            for p1 in -edge1:N1+edge1, p2 in -edge2:N2+edge2
        ]
        for I in CartesianIndices(al)
            p1 = I[1] - edge1-1; p2 = I[2] - edge2-1;
            if p1 < 0
                al[I] += ql_cross(p1/N1, p2/N2, 1,0)
            elseif p1 >= N1
                al[I] += ql_cross(p1/N1, p2/N2,-1,0)
            end
            if p2 < 0
                al[I] += ql_cross(p1/N1, p2/N2, 0,1)
            elseif p2 >= N2
                al[I] += ql_cross(p1/N1, p2/N2, 0,-1)
            end
            if al[I]<-π
                al[I] += 2π
            elseif al[I]>π
                al[I] -= 2π
            end
        end
        (ax, hm) = heatmap(fig[1,1], -edge1:N1+edge1, -edge2:N2+edge2, al,
            colormap = phasecolor, colorrange = (-pi,pi),
        )
        Colorbar(fig[1,2], hm, ticks = (-pi:0.5pi:pi, ["-π","-π/2","0","π/2","π"]))

        vlines!(ax, 0, color=:white)
        vlines!(ax, N1, color=:white)
        hlines!(ax, 0, color=:white)
        hlines!(ax, N2, color=:white)
        scatter!(ax, k1, k2, marker=:diamond)
        display(fig)
    end
    function plot_Fock_abs(k1, k2; N1 = N1, N2 = N2,
        edge1 = round(Int64, N1/2), edge2 = round(Int64, N2/2),)

        println("Fock[p1=$k1,p2=$k2,k1=$k1,k2=$k2,1,1,1,1,1,2] = ", Fock[1+k1,1+k2,1+k1,1+k2,1,1,1,1,1,2])
        fig = Figure(size=(1000,900));
        al = [ abs(Fock[1+(p1+N1)%N1, 1+(p2+N2)%N2,1+k1,1+k2,1,1,1,1,1,2])
            for p1 in -edge1:N1+edge1, p2 in -edge2:N2+edge2
        ]
        (ax, hm) = heatmap(fig[1,1], -edge1:N1+edge1, -edge2:N2+edge2, al,
            colorrange = (0.0,maximum(al)+0.01),
        )
        Colorbar(fig[1,2], hm, )

        vlines!(ax, 0, color=:white)
        vlines!(ax, N1, color=:white)
        hlines!(ax, 0, color=:white)
        hlines!(ax, N2, color=:white)
        scatter!(ax, k1, k2, marker=:diamond)
        display(fig)
    end
    # plotting hamiltonian H[τn′ = 1, τn = 2] c†_1 c_2
    function plot_hamiltonian_phase(H = H_int(ρ); ρ=ρ,
        edge1 = round(Int64, N1/2), edge2 = round(Int64, N2/2),
        phasecolor = range(Makie.Colors.HSV(0,1,1), stop=Makie.Colors.HSV(360,1,1), length=90))

        fig = Figure(size=(1000,900));
        al = [ angle(H[1+(k1+N1)%N1, 1+(k2+N2)%N2,1,1,1,2])
            for k1 in -edge1:N1+edge1, k2 in -edge2:N2+edge2
        ]
        for I in CartesianIndices(al)
            k1 = I[1] - edge1-1; k2 = I[2] - edge2-1;
            if k1 < 0
                al[I] += ql_cross(k1/N1, k2/N2,-1,0)
            elseif k1 >= N1
                al[I] += ql_cross(k1/N1, k2/N2,1,0)
            end
            if k2 < 0
                al[I] += ql_cross(k1/N1, k2/N2, 0,-1)
            elseif k2 >= N2
                al[I] += ql_cross(k1/N1, k2/N2, 0,1)
            end
            if al[I]<-π
                al[I] += 2π
            elseif al[I]>π
                al[I] -= 2π
            end
        end
        (ax, hm) = heatmap(fig[1,1], -edge1:N1+edge1, -edge2:N2+edge2, al,
            colormap = phasecolor, colorrange = (-pi,pi),
        )
        Colorbar(fig[1,2], hm, )

        vlines!(ax, 0, color=:white)
        vlines!(ax, N1, color=:white)
        hlines!(ax, 0, color=:white)
        hlines!(ax, N2, color=:white)

        scatter!(ax, 0.25N1, 0.75N2, marker=:diamond, color=:white)
        scatter!(ax, 0.75N1, 0.25N2, marker=:diamond, color=:white)

        scatter!(ax, N1/3, N2/3, marker=:diamond, color=:white)
        scatter!(ax, 2N1/3, 2N2/3, marker=:diamond, color=:white)

        display(fig)
    end
    function plot_hamiltonian_abs(H = H_int(ρ); ρ=ρ,
        edge1 = round(Int64, N1/2), edge2 = round(Int64, N2/2),
        phasecolor = range(Makie.Colors.HSV(0,1,1), stop=Makie.Colors.HSV(360,1,1), length=90))

        fig = Figure(size=(1000,900));
        al = [ abs(H[1+(k1+N1)%N1, 1+(k2+N2)%N2,1,1,1,2])./W0
        for k1 in -edge1:N1+edge1, k2 in -edge2:N2+edge2
        ]
        (ax, hm) = heatmap(fig[1,1], -edge1:N1+edge1, -edge2:N2+edge2, al,
            
        )
        Colorbar(fig[1,2], hm, )

        vlines!(ax, 0, color=:white)
        vlines!(ax, N1, color=:white)
        hlines!(ax, 0, color=:white)
        hlines!(ax, N2, color=:white)

        scatter!(ax, 0.25N1, 0.75N2, marker=:diamond, color=:white)
        scatter!(ax, 0.75N1, 0.25N2, marker=:diamond, color=:white)

        scatter!(ax, N1/3, N2/3, marker=:diamond, color=:white)
        scatter!(ax, 2N1/3, 2N2/3, marker=:diamond, color=:white)

        display(fig)
    end
=#

end