"""
General Hartree-Fock algorithm that preserves momentum and mix multiple bands
using iteration
"""
module HartreeFock

@kwdef mutable struct HFPara

end


# only T=0
function hf_onestep!(new_rho, ρ; 
    H0, Hint, step_procession=[],
    ) 
    # H[k1, k2, τn′, τn]
    # ρ[k1, k2, τ, τ′]
    H = Hint + H0

    bands = zeros(Float64, N1, N2, 2)
    for k1 in axes(H,1), k2 in axes(H,2)
        vals, vecs = eigen(Hermitian(H[k1,k2,:,:]) )
        new_rho[k1,k2,:,:] .= vecs[:,1] * vecs[:,1]'
    end

    for f in step_procession
        f(new_rho)
    end

    return
end
function band(ρ; para::LLHFNumPara, Hint = hf_interaction(ρ,para), )
    H = Hint + para.H0
    N1 = para.N1; N2 = para.N2
    band = zeros(Float64, N1, N2, 2)
    for k1 in 0:N1-1, k2 in 0:N2-1
        vals, vecs = eigen(Hermitian(H[1+k1,1+k2,:,:]) )
        band[1+k1, 1+k2, :] .= vals
    end
    return band
end
function hf_converge!(ρ;
    para::LLHFNumPara, EPA, HINT, OneStep,

    procession_stpes = 0, procession = [], 
    error_tolerance = 1E-8, iterations = 1:200,
    mixing = false, mixing_rate = 0.5,
    stepwise_output::Bool = false, final_output::Bool = true,
    )


    if complusive_mixing
        println("Iteration uses complusive mixing rate $complusive_mixing_rate.")
    end
    
    hint = hf_interaction(ρ, para)

    new_rho = similar(ρ)
    new_hint = similar(hint)

    for i in 1:max_iter_times
        if i == post_process_times+1
            empty!(post_procession)
        end
        hf_onestep!(new_rho, ρ; Hint = hint, 
            post_procession = post_procession, para = para
        )
        
        new_hint .= hf_interaction(new_rho, para)

        error = maximum(abs.(new_rho-ρ))

        E0 = EPA(ρ, Hint = hint, para=para, imag_print=false)
        stepwise_output && println("$i \t DM error:$error \t E/S:$E0")
        E1 = EPA(new_rho, Hint = new_hint, para=para, imag_print=false)
        Eh = EPA((ρ+new_rho)/2., para=para, imag_print=false) # h stands for halfway
        k = E0+E1-2Eh

        if complusive_mixing
            if complusive_mixing_rate == 0.5
                stepwise_output && println("update rate = 0.5: unstable solution with higher energy")
                ρ .= (ρ+new_rho)/2.
                hint = hf_interaction(ρ, para)
            else
                x = complusive_mixing_rate
                stepwise_output && println("update rate = $(round(x; digits=2))")
                ρ .+= x*(new_rho - ρ)
                hint = hf_interaction(ρ, para)
            end
        else
            if k <= 0. 
                if E0 >= E1
                stepwise_output && println("update rate = 1.0: unstable solution with lower energy")
                ρ .= new_rho
                hint .= new_hint
                else
                    stepwise_output && println("update rate = 0.5: unstable solution with higher energy")
                    ρ .= (ρ+new_rho)/2.
                    hint = hf_interaction(ρ, para)
                end
            elseif k <= 2(E0-E1)
                stepwise_output && println("update rate = 1.0: energy minimum is ahead but we constrained to 1.0")
                ρ .= new_rho
                hint .= new_hint
            else
                x0 = 0.5 + (E0-E1)/k
                x = max(0.1, x0)
                stepwise_output && println("update rate = $(round(x; digits=2)): energy minimum at $(round(x0; digits=2))")
                ρ .+= x*(new_rho - ρ)
                hint = hf_interaction(ρ, para)
            end
        end


        if error <= error_tolerance
            final_output && println("converged in $i iterations, density error = $error")
            break
        end
        if i == max_iter_times
            final_output && println("not converged after $max_iter_times iterations, density error = $error.")
        end
    end

    return ρ
end
function HF_solve_iteration(para, ρ;
    initial_procession=[], final_procession=[],
    iter...
    )

    for f in initial_procession
        f(ρ) 
    end
    
    ρ = hf_converge!(ρ; para = para, iter...)

    for f in final_procession
        f(ρ) 
    end

    return ρ
end



end