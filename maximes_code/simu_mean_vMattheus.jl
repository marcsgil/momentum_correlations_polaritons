using CairoMakie, ProgressMeter, FFTW, LinearAlgebra, DSP

#Function taking into account the non-linearity of the fluid
function directSpaceInteraction!(array, potential_lp, g_lp, δt, δ, NoisePumpVector)
    @. array = array * cis(-δt * (potential_lp + g_lp * (abs2(array) - 1 / δx))) + NoisePumpVector
end

function step!(ϕ_lp, ϕ_lp_fft, matrix_fft, matrix_ifft, losses_and_effective_detuning, potential_lp, g_lp, δt, δx, NoisePumpVector)
    mul!(ϕ_lp_fft, matrix_fft, ϕ_lp)
    ϕ_lp_fft .*= cis.(-δt * losses_and_effective_detuning)
    mul!(ϕ_lp, matrix_ifft, ϕ_lp_fft)
    directSpaceInteraction!(ϕ_lp, potential_lp, g_lp, δt, δx, NoisePumpVector)
end

#Function simulating the evolution of the system until the stationary state
function SimulationSteadyState(matrix_fft, matrix_ifft, losses_and_effective_detuning, potential_lp, g_lp, δt, δx, NoisePumpVector, n_stationary, n_saves)
    ϕ_lp = zero(NoisePumpVector)
    ϕ_lp_fft = zero(NoisePumpVector)

    result = [ϕ_lp]

    save_every = n_stationary ÷ n_saves

    @showprogress for step = 1:n_stationary
        step!(ϕ_lp, ϕ_lp_fft, matrix_fft, matrix_ifft, losses_and_effective_detuning, potential_lp, g_lp, δt, δx, NoisePumpVector)
        if step % save_every == 0
            push!(result, copy(ϕ_lp))
        end
    end

    stack(result)
end

heaviside(x) = x ≥ 0

function pump(x, param)
    F_p_max = param.F_p_max
    F_p_support_u = param.F_p_support_u
    F_p_support_d = param.F_p_support_d
    k_p_u = param.k_p_u
    k_p_d = param.k_p_d
    x_pump_end = param.x_pump_end
    σ_sech = param.σ_sech

    (((F_p_max - F_p_support_u) * sech(x / σ_sech) + F_p_support_u * heaviside(x_pump_end - x)) * cis(k_p_u * x)
     +
     F_p_support_d * heaviside(x - x_pump_end) * cis(k_p_d * x))
end

function potential(x, param)
    high_absorbing = param.high_absorbing
    σ_absorbing = param.σ_absorbing
    x_final = param.x_final
    x_defect = param.x_defect

    -im * high_absorbing * exp(-((x - x_final)^2) / σ_absorbing^2) + height_defect * exp.(-((x - x_defect)^2) / σ_defect^2)
end

function dispersion(k, param)
    γ_lp = param.γ_lp
    ħ = param.ħ
    m_lp = param.m_lp
    δ₀ = param.δ₀

    -im * γ_lp / 2 + ħ * k^2 / 2m_lp - δ₀
end


#Values of fondamental parameters of system
ħ = 0.6582 #meV.ps
γ_lp = 0.047 / ħ
g_lp = 0.0003 / ħ
m_lp = ħ^2 / (2 * 1.29)

# #Definition of time grid
δt = 5e-2
T_ss = 8000
n_stationary = floor(T_ss / δt)

#Definition of space grid
x_final = 2000
n_x = 2048
δx = x_final / n_x
δk = 2 * π / x_final
x_vector = range(; start=0, step=δx, length=n_x)
#k_vector = fftshift(LinRange(-(n_x / 2 - 1) * δk, n_x / 2 * δk, n_x))
k_vector = fftfreq(n_x, 2π / δx)


#Definition of the potential reigning in the cavity
high_absorbing = 4.5 / ħ
σ_absorbing = 20
x_defect = 400
height_defect = -0.85 / ħ
σ_defect = 0.69

#Definition of the pump parameters
δ₀ = 0.49 / ħ
k_p_u = 0.27
k_p_d = 0.539

#Effect on the polariton cavity
effective_detuning_u = ħ * δ₀ - ħ^2 * k_p_u^2 / (2 * m_lp)
c_sonic = sqrt(effective_detuning_u / m_lp)
F_p_sonic = sqrt((((effective_detuning_u - m_lp * c_sonic^2) / ħ)^2 + (γ_lp / 2)^2) * m_lp * c_sonic^2 / (ħ * g_lp))

effective_detuning_d = ħ * δ₀ - ħ^2 * k_p_d^2 / (2 * m_lp)
c_sonic_d = sqrt(effective_detuning_d / m_lp)
F_p_sonic_d = sqrt((((effective_detuning_d - m_lp * c_sonic_d^2) / ħ)^2 + (γ_lp / 2)^2) * m_lp * c_sonic_d^2 / (ħ * g_lp))

#Spatial properties
F_p_support_u = F_p_sonic + 0.01  # intensity to support the upstream polariton where we want in the upper branch; in meV/ps
F_p_support_d = F_p_sonic_d + 0.13# values found by Malte in 2023
F_p_max = 9
x_pump_end = x_defect - 7
σ_sech = 20

param = (; F_p_max, F_p_support_u, F_p_support_d, k_p_u, k_p_d, x_pump_end, σ_sech,
    high_absorbing, σ_absorbing, x_final, x_defect, height_defect, σ_defect, 
    γ_lp, ħ, m_lp, g_lp, δ₀)

E_p = [pump(x, param) for x in x_vector]
NoisePumpVector = -1im * δt * E_p

potential_lp = [potential(x, param) for x in x_vector]

#Definition of FFT quantities
matrix_fft = plan_fft(zeros(ComplexF64, n_x), flags=FFTW.ESTIMATE, timelimit=Inf)
matrix_ifft = plan_ifft(zeros(ComplexF64, n_x), flags=FFTW.ESTIMATE, timelimit=Inf)

#Term of detuning and losses in GP equation
losses_and_effective_detuning = [dispersion(k, param) for k in k_vector]


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