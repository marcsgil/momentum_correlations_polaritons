using CairoMakie, ProgressMeter, FFTW, LinearAlgebra, DSP

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
x_vector = LinRange(0, x_final - δx, n_x)
k_vector = fftshift(LinRange(-(n_x / 2 - 1) * δk, n_x / 2 * δk, n_x))

#Definition of the potential reigning in the cavity
high_absorbing = 4.5 / ħ
σ_absorbing = 20
x_defect = 400
height_defect = -0.85 / ħ
σ_defect = 0.69
potential_lp = -1im * high_absorbing * exp.(-((x_vector .- x_final) .^ 2) / σ_absorbing^2) .+ height_defect * exp.(-((x_vector .- x_defect) .^ 2) / σ_defect^2)

with_theme(theme_latexfonts()) do
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel="x [μm]", ylabel="Potential [meV]", title="Potential in the cavity")
    lines!(ax, x_vector .- x_defect, real(potential_lp), color=:blue)
    fig
end

#Definition of the pump parameters
δ₀ = 0.49 / ħ
k_p_u = 0.27
k_p_d = 0.539#commented are values found by Malte in 2023

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

function heaviside(vector)
    for i ∈ eachindex(vector)
        if vector[i] < 0
            vector[i] = 0.
        else
            vector[i] = 1.
        end
    end
    return vector
end

E_p = ((F_p_max - F_p_support_u) * sech.(x_vector / σ_sech) + (F_p_support_u .* heaviside(collect(x_pump_end .- x_vector)))) .* exp.(1im * k_p_u * x_vector) + F_p_support_d .* heaviside(collect(x_vector .- x_pump_end)) .* exp.(1im * k_p_d * x_vector)

x_beg_u = 70
x_end_u = 365
x_beg_d = 401
x_end_d = 745

with_theme(theme_latexfonts()) do
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel="x [μm]", ylabel="Pump Ep [a.u.]", title="Pumping profile in the cavity")
    lines!(ax, x_vector .- x_defect, abs.(E_p), color=:blue)
    fig
end

#Definition of FFT quantities
normalization_ftt = sqrt(x_final) / δx
matrix_fft = plan_fft(zeros(ComplexF64, n_x), flags=FFTW.ESTIMATE, timelimit=Inf)
matrix_ifft = plan_ifft(zeros(ComplexF64, n_x), flags=FFTW.ESTIMATE, timelimit=Inf)
matrix_fft2 = plan_fft(zeros(ComplexF64, (n_x, n_x)), flags=FFTW.ESTIMATE, timelimit=Inf)

#Term of detuning and losses in GP equation
losses_and_effective_detuning = @. cis(-δt * (-im * γ_lp / 2 + ħ * k_vector^2 / 2m_lp - δ₀))

#Function taking into account the pump and the fluctuations effects
NoisePumpVector = -1im * δt * E_p


print("Go to simulation")