"""
General Hartree-Fock algorithm that preserves momentum and mix multiple bands
using self-consistent iteration
"""
module HartreeFock

    using MKL, LinearAlgebra


    # one step update functions
    begin

        # T=0, integer band filling
        function HF_onestep_T0_int!(new_rho, ρ, H0;
            Hint, step_processions=[], filling::Int64,
            parameter_dim = size(H0)[begin+2:end],
            band_num = size(H0)[begin],
            )

            # H[τn′, τn, prmts]
            # ρ[τ, τ′, prmts]
            H = Hint + H0

            new_rho .= 0.0
            bands = zeros(Float64, band_num, parameter_dim...)
            for paraIdx in CartesianIndices(parameter_dim)
                bands[:, paraIdx] , vecs = eigen(Hermitian(view(H, :, :, paraIdx)) )
                for i_band in 1:filling
                    new_rho[:, :, paraIdx] .+= vecs[:,i_band] * vecs[:,i_band]'
                end
            end

            for f! in step_processions
                f!(new_rho)
            end

            return bands
        end


    end


    # solving functions
    begin


        """
        Hartree-Fock self-consistent iteration with dynamic mixing.
        The mixing rate is determined based on energy.
        input:
        EPA: Function to calculate energy per area 
        HFInteraction: Function to calculate Hartree-Fock interaction
        HFOneStep: Function to perform one iteration step
        error_tolerance: Convergence criteria of density matrix elements
        iteration_steps: Maximum number of iteration steps
        procession_steps: maximum number of iteration steps with processions
        step_processions: List of functions for processions
        stepwise_output: Flag to enable stepwise output
        process_output: Flag to enable outputs of starting and ending self-consistent process
        """
        function HF_SC_mixing!(ρ, H0; filling::Int64,
            mixing_rate::Union{Symbol, Float64}=:dynamic,
            EPA, HFInteraction, HFOneStep! = HF_onestep_T0_int!,
            error_tolerance = 1E-8, iteration_steps = 200,
            procession_steps = iteration_steps, step_processions = [],
            stepwise_output::Bool = false, process_output::Bool = true,
            )

            if mixing_rate == :dynamic
                process_output && println("Iteration uses dynamic mixing rate.")
            elseif mixing_rate isa Float64 && mixing_rate > 0
                process_output && println("Iteration uses fixed mixing rate $mixing_rate")
            else
                error("Invalid mixing rate: $mixing_rate. Must be :dynamic or a positive Float64.")
            end
            

            hint = HFInteraction(ρ)
            new_rho = similar(ρ)
            new_hint = similar(hint)

            for i in 1:iteration_steps

                if i == procession_steps + 1
                    empty!(step_processions)
                end


                HFOneStep!(new_rho, ρ, H0;
                    Hint = hint, step_processions = step_processions, filling = filling
                )
                new_hint .= HFInteraction(new_rho)
                error = maximum(abs.(new_rho-ρ))

                if mixing_rate == :dynamic
                    E0 = EPA(ρ, hint)
                    stepwise_output && println("$i \t DM error:$error \t E/S:$E0")
                    E1 = EPA(new_rho, new_hint)
                    Eh = EPA((ρ+new_rho)/2.) # h stands for halfway
                    k = E0+E1-2Eh
                    if k <= 0. 
                        if E0 >= E1
                        stepwise_output && println("update rate = 1.0: unstable solution with lower energy")
                        ρ .= new_rho
                        hint .= new_hint
                        else
                            stepwise_output && println("update rate = 0.5: unstable solution with higher energy")
                            ρ .= (ρ+new_rho)/2.
                            hint = HFInteraction(ρ)
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
                        hint = HFInteraction(ρ)
                    end
                else
                    E0 = EPA(ρ, hint)
                    stepwise_output && println("$i \t DM error:$error \t E/S:$E0")
                    stepwise_output && println("update rate = $(round(mixing_rate; digits=4))")
                    ρ .+= mixing_rate*(new_rho - ρ)
                    hint = HFInteraction(ρ)
                end

                if error <= error_tolerance
                    process_output && println("converged in $i iterations, density error = $error")
                    break
                end
                if i == iteration_steps
                    process_output && println("not converged after $iteration_steps iterations, density error = $error.")
                end

                
            end

            return
        end



        """
        Hartree-Fock self-consistent iteration with fixed mixing.
        input:
        EPA: Function to calculate energy per area 
        HFInteraction: Function to calculate Hartree-Fock interaction
        HFOneStep: Function to perform one iteration step
        error_tolerance: Convergence criteria of density matrix elements
        iteration_steps: Maximum number of iteration steps
        procession_steps: maximum number of iteration steps with processions
        step_processions: List of functions for processions
        stepwise_output: Flag to enable stepwise output
        process_output: Flag to enable outputs of starting and ending self-consistent process
        """
        function HF_SC_fixed_mixing!(ρ, H0; filling, mixing_rate=1.0,
            EPA, HFInteraction, HFOneStep! = HF_onestep_T0_int!,
            error_tolerance = 1E-8, iteration_steps = 200,
            procession_steps = iteration_steps, step_processions = [],
            stepwise_output::Bool = false, process_output::Bool = true,
            )

            if process_output
                println("Iteration uses fixed mixing rate $mixing_rate.")
            end

            hint = HFInteraction(ρ)
            new_rho = similar(ρ)
            new_hint = similar(hint)

            for i in 1:iteration_steps

                if i == procession_steps + 1
                    empty!(step_processions)
                end


                HFOneStep!(new_rho, ρ, H0;
                    Hint = hint, step_processions = step_processions, filling = filling
                )
                new_hint .= HFInteraction(new_rho)
                error = maximum(abs.(new_rho-ρ))


                E0 = EPA(ρ, hint)
                stepwise_output && println("$i \t DM error:$error \t E/S:$E0")
                stepwise_output && println("update rate = $(round(mixing_rate; digits=4))")
                ρ .+= mixing_rate*(new_rho - ρ)
                hint = HFInteraction(ρ)

                

                if error <= error_tolerance
                    process_output && println("converged in $i iterations, density error = $error")
                    break
                end
                if i == iteration_steps
                    process_output && println("not converged after $iteration_steps iterations, density error = $error.")
                end

                
            end

            return
        end







    end


    function HF_solve_method(ρ, H0;
        initial_procession=[], final_procession=[],
        iter_method! = :HF_SC_mixing!, iter_kwargs...
        )

        for fi! in initial_procession
            fi!(ρ) 
        end

        eval(iter_method!)(ρ, H0; iter_kwargs...)

        for ff! in final_procession
            ff!(ρ) 
        end

        return ρ
    end

end