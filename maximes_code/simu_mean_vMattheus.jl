#Function taking into account the non-linearity of the fluid
function PotentialInteractionsAction!(array, potential_lp, g_lp, δt, δx)
    @. array *= cis(-δt * (potential_lp + g_lp * (abs2(array) - 1 / δx)))
end

function PumpAction!(array, NoisePumpVector)
    array .+= NoisePumpVector
end

#Function using the different terms in GP equation to simulate a complete evolution

function OneEvolutionNoNoise!(ϕ_lp, ϕ_lp_fft, matrix_fft, matrix_ifft, losses_and_effective_detuning, potential_lp, g_lp, δt, δx, NoisePumpVector)
    mul!(ϕ_lp_fft, matrix_fft, ϕ_lp)
    ϕ_lp_fft .*= losses_and_effective_detuning
    mul!(ϕ_lp, matrix_ifft, ϕ_lp_fft)
    PotentialInteractionsAction!(ϕ_lp, potential_lp, g_lp, δt, δx)
    PumpAction!(ϕ_lp, NoisePumpVector)
end

#Function simulating the evolution of the system until the stationary state
function SimulationSteadyState(matrix_fft, matrix_ifft, losses_and_effective_detuning, potential_lp, g_lp, δt, δx, NoisePumpVector, n_stationary, n_saves)
    ϕ_lp = zero(NoisePumpVector)
    ϕ_lp_fft = zero(NoisePumpVector)

    result = [ϕ_lp]

    save_every = n_stationary ÷ n_saves

    @showprogress for step = 1:n_stationary
        # println("Step $step: Starting computation...") # useful for debugging, be careful when doing long simulations
        OneEvolutionNoNoise!(ϕ_lp, ϕ_lp_fft, matrix_fft, matrix_ifft, losses_and_effective_detuning, potential_lp, g_lp, δt, δx, NoisePumpVector)
        if step % save_every == 0
            push!(result, copy(ϕ_lp))
        end
    end

    stack(result)
end


result = SimulationSteadyState(matrix_fft, matrix_ifft, losses_and_effective_detuning, potential_lp, g_lp, δt, δx, NoisePumpVector, n_stationary, 1024)

heatmap(abs2.(result))
##
ϕ_steady = result[:, end]
with_theme(theme_latexfonts()) do
    x_beg_u = 70
    x_end_u = 365
    x_beg_d = 401
    x_end_d = 745
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel="x [μm]", ylabel="velocities [μm/ps]", title="Velocities of the meanfield")
    lines!(x_vector .- x_defect, sqrt.(ħ * g_lp * abs2.(ϕ_steady) / m_lp), label="c_x")
    lines!(x_vector[2:n_x-1] .- x_defect, ħ / m_lp * diff(unwrap(angle.(ϕ_steady[2:n_x]))) / δx, label="v_x")
    xlims!(ax, [x_beg_u - x_defect, x_end_d - x_defect])
    ylims!(ax, [0, 5])
    fig
end