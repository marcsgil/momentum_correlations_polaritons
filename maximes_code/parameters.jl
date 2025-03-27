plot_potential = false
plot_pumping = true
plot_window = false
save = true

folder="/home/stagios/Marcos/momentum_correlations_polaritons/maximes_code/"
# folder = "./"



# #Definition of time grid
const δt = 0.5 * 0.01
const T_ss = 12000
const n_stationary = floor(T_ss / δt)

const number_realization_simu = 5000
const n_sample_simu = 500
const δt_sample_simu = n_sample_simu * δt
const tsimu_final = T_ss + number_realization_simu * δt_sample_simu
const n_time_simu = floor(tsimu_final / δt)

const number_realization_bogo = 5000
const n_sample_bogo = 150
const δt_sample_bogo = n_sample_bogo * δt
const tbogo_final = T_ss + number_realization_bogo * δt_sample_bogo
const n_time_bogo = floor(tbogo_final / δt)
const δω = 2 * π / (number_realization_bogo * δt_sample_bogo)

function omeg_to_n(o)
    return (floor(Int64, number_realization_bogo / 2 - o / δω))
end
function n_to_omeg(n)
    return ((number_realization_bogo / 2 - 1) * δω - (n - 1) * δω)
end

#Definition of space grid
const x_final = 2000
const n_x = 2048
const δx = x_final / n_x
const δk = 2 * π / x_final
const x_vector = LinRange(0, x_final - δx, n_x)
const relation_TWA_valid = γ_lp * δx / g_lp
const k_vector = fftshift(LinRange(-(n_x / 2 - 1) * δk, n_x / 2 * δk, n_x))
const k_vect = LinRange(-(n_x / 2 - 1) * δk, n_x / 2 * δk, n_x)
const ω_lp_vector = ω_lp_0 .+ ħ * k_vector .^ 2 / (2 * m_lp)

function n_to_k(n)
    return (-(n_x / 2 - 1) * δk + (n - 1) * δk)
end
function k_to_n(k)
    return (floor(Int64, n_x / 2 + k / δk))
end

#Definition of the potential reigning in the cavity
const high_absorbing = 4.5 / ħ
const σ_absorbing = 20
const x_defect = 400
const height_defect = -0.85 / ħ
const σ_defect = 0.69
const potential_lp = -1im * high_absorbing * exp.(-((x_vector .- x_final) .^ 2) / σ_absorbing^2) .+ height_defect * exp.(-((x_vector .- x_defect) .^ 2) / σ_defect^2)


#Definition of the pump parameters
const ω_p = 1473.85 / ħ
const k_p_u = 0.27
const k_p_d = 0.539#commented are values found by Malte in 2023

#Effect on the polariton cavity
const effective_detuning_u = ħ * ω_p - ħ * (ω_lp_0 + ħ * k_p_u^2 / (2 * m_lp))
const c_sonic = sqrt(effective_detuning_u / m_lp)
const F_p_sonic = sqrt((((effective_detuning_u - m_lp * c_sonic^2) / ħ)^2 + (γ_lp / 2)^2) * m_lp * c_sonic^2 / (ħ * g_lp))

const effective_detuning_d = ħ * ω_p - ħ * (ω_lp_0 + ħ * k_p_d^2 / (2 * m_lp))
const c_sonic_d = sqrt(effective_detuning_d / m_lp)
const F_p_sonic_d = sqrt((((effective_detuning_d - m_lp * c_sonic_d^2) / ħ)^2 + (γ_lp / 2)^2) * m_lp * c_sonic_d^2 / (ħ * g_lp))

#Spatial properties
const F_p_support_u = F_p_sonic + 0.01  # intensity to support the upstream polariton where we want in the upper branch; in meV/ps
const F_p_support_d = F_p_sonic_d + 0.13# values found by Malte in 2023
const F_p_max = 9
const x_pump_end = x_defect - 7
const σ_sech = 20

const E_p = ((F_p_max - F_p_support_u) * sech.(x_vector / σ_sech) + (F_p_support_u .* heaviside(collect(x_pump_end .- x_vector)))) .* exp.(1im * k_p_u * x_vector) + F_p_support_d .* heaviside(collect(x_vector .- x_pump_end)) .* exp.(1im * k_p_d * x_vector)

if plot_pumping
    figure()
    title("Pumping profile in the cavity")
    plot(x_vector .- x_defect, abs.(E_p))
    xlabel("x [μm]")
    ylabel("Pump Ep [a.u.]")
    ylim([0, 9])
    if save
        savefig(folder * "pumping")
    end
    # figure(figsize=(8, 1))
    # plot(x_vector .- x_defect,abs.(E_p))
    # box(true)  # Remove the box around the plot
    # xticks([])   # Remove x-axis ticks
    # #yticks([])   # Remove y-axis ticks
    # ylim([0, 2])
    # xlim([x_beg_u-x_defect,x_end_d-x_defect])
    # ylabel("|Ep|")
    # legend().set_visible(false)
end

if plot_potential
    open(folder * "pump_en.txt", "w") do io
        @inbounds for i = 1:n_x
            writedlm(io, abs.(E_p)[i])
        end
    end
    open(folder * "x_vector.txt", "w") do io
        @inbounds for i = 1:n_x
            writedlm(io, x_vector[i])
        end
    end
    open(folder * "potential.txt", "w") do io
        @inbounds for i = 1:n_x
            writedlm(io, real(potential_lp[i]))
        end
    end
end

#Definition of the probe
const F_s = F_p_support_u / 1000
const x_s = 300
const σ_s = 5.0
const k_s = 0.33

function resdisp(reg::Int64, branch::Int64, ks::Float64)
    if reg == 0 #upstream
        if branch == -1 #ghost
            ωs = v_u * (ks - k_u) - √(abs.((ħ * (ks - k_u)^2 / (2 * m_lp) + g_lp * density_u - detuning_u / ħ) * (ħ * (ks - k_u)^2 / (2 * m_lp) + 3 * g_lp * density_u - detuning_u / ħ)))
        else #normal
            ωs = v_u * (ks - k_u) + √(abs.((ħ * (ks - k_u)^2 / (2 * m_lp) + g_lp * density_u - detuning_u / ħ) * (ħ * (ks - k_u)^2 / (2 * m_lp) + 3 * g_lp * density_u - detuning_u / ħ)))
        end
    else #downstream
        if branch == -1
            ωs = v_d * (ks - k_d) - √(abs.((ħ * (ks - k_d)^2 / (2 * m_lp) + g_lp * density_d - detuning_d / ħ) * (ħ * (ks - k_d)^2 / (2 * m_lp) + 3 * g_lp * density_d - detuning_d / ħ)))
        else
            ωs = v_d * (ks - k_d) + √(abs.((ħ * (ks - k_d)^2 / (2 * m_lp) + g_lp * density_d - detuning_d / ħ) * (ħ * (ks - k_d)^2 / (2 * m_lp) + 3 * g_lp * density_d - detuning_d / ħ)))
        end
    end
    return (ωs)
end
function inv_resdisp(reg::Int64, branch::Int64, os::Float64)
end

function resonance(bogo::Vector{Float64}, k_s::Float64)
    n_k_s = k_to_n(k_s)
    ω_s = bogo[n_k_s]
    return (ω_s)
end
function resonance_down(bogo::Vector{Float64}, ω_s::Float64)
    i = k_to_n(k_p_d) + argmin(abs.(abs.(bogo) .- ω_s)[k_to_n(k_p_d):end]) - 1
    return (k_vect[i])
end
function conj_resonance(bogo::Vector{Float64}, ω_s::Float64)
    i = argmin(abs.(bogo .- ω_s)[1:k_to_n(k_p_u)])
    return (k_vect[i])
end
function conj_resonance_down(bogo::Vector{Float64}, ω_s::Float64)
    i = argmin(abs.(bogo .+ ω_s)[1:k_to_n(k_p_d)])
    return (k_vect[i])
end
function probe(xs::Int64, σs::Float64, k_s::Float64)
    return F_s * exp.(-(x_vector .- xs) .^ 2 / σs^2) .* exp.(1im * k_s * x_vector) #.* door(length(x_vector),floor(Int64,(xs-σ_s-0.5*σs)/δx),floor(Int64,(xs+σs+0.5*σs)/δx))
end

const t_ss_probe = 400
const n_ss_probe = floor(t_ss_probe / δt)
const number_realization_ω = 3000
const n_sample_probe = floor(Int64, n_sample_bogo)
const δt_sample_probe = n_sample_probe * δt
const δω_probe = 2 * π / (number_realization_ω * δt_sample_probe)

function omeg_to_n_pr(o)
    return (floor(Int64, number_realization_ω / 2 - o / δω_probe))
end
function n_to_omeg_pr(n)
    return ((number_realization_ω / 2 - 1) * δω_probe - (n - 1) * δω_probe)
end

#Definition of the homogeneous zone
const x_beg_u = 70
const x_end_u = 365
const x_beg_d = 401
const x_end_d = 745

#Definition of the window
const σ_u = 300
const x1_u = 70
const x2_u = x1_u + σ_u

const σ_d = 250
const x1_d = 401
const x2_d = x1_d + σ_d

if plot_window
    figure()
    plot(x_vector, wind_hanning(zeros(n_x) .+ 1, x1_u, x2_u), label="upstream window", "y--", alpha=0.25)
    plot(x_vector, wind_hanning(zeros(n_x) .+ 1, x1_d, x2_d), label="downstream window", "y--", alpha=0.25)
    plot(x_vector, abs.(probe(x_s, σ_s, k_s)) ./ maximum(abs.(probe(x_s, σ_s, k_s))), label="probe", "g", alpha=0.8)
    plot(x_defect .+ zeros(3), LinRange(0, 1, 3), label="defect", "r--", alpha=0.25)
    xlabel("x en μm")
    ylabel("[a.u.]")
    title("x_s=$x_s; σ_s=$σ_s")
    legend()
    if save
        savefig(folder * "fenêtrage")
    end
end

#Size of the box near the horizon in which the correlations will be computed
const ℓ_correlation_real_d = 115
const ℓ_correlation_real_u = 115
const n_correlation_real_start = floor(Int, (x_defect - ℓ_correlation_real_u) / δx)
const n_correlation_real_end = floor(Int, (x_defect + ℓ_correlation_real_d) / δx)

const EVERY = 1e5

#Definition of FFT quantities
const normalization_ftt = sqrt(x_final) / δx
const matrix_fft = plan_fft(zeros(ComplexF64, n_x), flags=FFTW.ESTIMATE, timelimit=Inf)
const matrix_ifft = plan_ifft(zeros(ComplexF64, n_x), flags=FFTW.ESTIMATE, timelimit=Inf)
const matrix_fft2 = plan_fft(zeros(ComplexF64, (n_x, n_x)), flags=FFTW.ESTIMATE, timelimit=Inf)

#Term of detuning and losses in GP equation
const losses_and_effective_detuning = exp.(-1im * δt * (ω_lp_vector .- ω_p .- 1im * γ_lp / 2))


#Function taking into account the pump and the fluctuations effects
const NoisePumpVector = -1im * δt * E_p
const diffusion_FokkerPlanck = 0.5 * sqrt.(δt * (γ_lp .+ 2 * abs.(imag.(potential_lp))) / δx)


print("Go to simulation")