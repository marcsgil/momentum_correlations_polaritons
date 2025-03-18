#using IJulia
using Distributed
using Printf
using Statistics
using FFTW
#using JLD2
#using BenchmarkTools
using StaticArrays
using Profile
using LinearAlgebra
#using DSP
#using DelimitedFiles
using Dates
using CairoMakie

#Gross-Pitaevskii equation considered : i*h_bar*psi_t = [h_bar*omega_lp_0 - h_bar^2/(2*m_lp)*Delta + h_bar*g_lp*|psi|^2]*psi - i*h_bar*gamma_lp/2*psi + h_bar*F_p
const ħ = 0.6582 # in meV.ps

#Values of fondamental parameters of system Nguyen
# const ω_lp_0 = 1473.36/ħ # [Nguyen et al.] (experimental); in 1/ps
# const γ_lp = 0.047/ħ # [Nguyen et al.] (experimental); in 1/ps
# const g_lp = 3e-4/ħ # [Nguyen et al.] g_lp = 0.0003/h_bar (experimental); [Grisins et al.] g_lp = 0.005/h_bar; in µm/ps
# const m_lp = ħ^2/(2*1.29) #in meV.ps^2/µm^2; [Nguyen et al.] m_lp = 3*10^(-5)*m_electron 

#Values of fondamental parameters of historic 2D cavity [Claude et al PRL 2021]
const ω_lp_0 = 1482.717/ħ # [from Kevin's data] (experimental); in 1/ps => meV
const γ_lp = 0.08/ħ # in 1/ps
const g_lp = 3e-4/ħ # [not sure as, in the experiment we measure gn and in 2D, I take Nguyen's value] in µm/ps
const m_lp = ħ^2/(2*1.29) #in meV.ps^2/µm^2; [Nguyen et al.] m_lp = 3*10^(-5)*m_electron 



#Definition of time grid
const number_realization_wished = 15874# that's 63*15874~&e6 realisations#4761905 that's ~3e8 realisations
const δt = 0.01
const n_sample = 500
const δt_sample = n_sample*δt
const T_ss = 4000
const t_final = T_ss + number_realization_wished*δt_sample
const n_time = floor(t_final/δt)
const n_time_sample = floor((t_final - T_ss)/δt_sample)


#Definition of space grid
const x_final = 800
const n_x = 2048
const δx = x_final/n_x
const δk = 2*π/x_final
const x_vector = LinRange(0, x_final - δx, n_x)
const relation_TWA_valid = γ_lp*δx/g_lp
#Should be commented part
const k_vector = fftfreq(n_x,1/δk) |> fftshift#const k_vector = fftshift(LinRange(-(n_x/2 - 1)*δk, n_x/2*δk, n_x))
const ω_lp_vector =  ω_lp_0 .+ ħ*k_vector.^2/(2*m_lp)


#Definition of the potential reigning in the cavity
const high_absorbing = 4.5/ħ
const σ_absorbing = 20
const x_defect = 300
const high_defect =-0.85/ħ
const σ_defect = 0.75
const potential_lp = -1im*high_absorbing*exp.(-((x_vector.-x_final).^2)/σ_absorbing^2) .+ high_defect*exp.(-((x_vector.-x_defect).^2)/σ_defect^2)


lines(abs.(potential_lp))

#Definition of the pump parameters
const ener_detu = 0.3/ħ #meV
const ω_p = ener_detu + ω_lp_0 # meV
# const ω_p = 1482.871/ħ # in 1/ps => meV
const k_p = 0.1
const effective_detuning = ħ*ω_p - ħ*(ω_lp_0 + ħ*k_p^2/(2*m_lp)) #Kévin says the detuning is about 0.17meV
const c_sonic = sqrt(effective_detuning/m_lp)
const F_p_sonic = sqrt((((effective_detuning - m_lp*c_sonic^2)/ħ)^2 + (γ_lp/2)^2)*m_lp*c_sonic^2/(ħ*g_lp))
const F_p_max = 5
const x_pump_end = x_final
const σ_sech = 20
function heaviside(vector)
    @inbounds for i = 1:length(vector)
        if vector[i] < 0
            vector[i] = 0.
        else
            vector[i] = 1.
        end
    end
    return vector
end
const k_p_d = 0.4
const effective_detuning_d = ħ*ω_p - ħ*(ω_lp_0 + ħ*k_p_d^2/(2*m_lp)) #Kévin says the detuning is about 0.17meV
const c_sonic_d = sqrt(effective_detuning_d/m_lp)
const F_p_sonic_d = sqrt((((effective_detuning_d - m_lp*c_sonic_d^2)/ħ)^2 + (γ_lp/2)^2)*m_lp*c_sonic_d^2/(ħ*g_lp))
function invertedheaviside(vector)
    @inbounds for i = 1:length(vector)
        if vector[i] <= x_defect
            vector[i] = 0.
        else
            vector[i] = 1.
        end
    end
    return vector
end
const E_p = ((F_p_max - F_p_sonic)*sech.(x_vector/σ_sech) .+ F_p_sonic*heaviside(collect(x_pump_end .- x_vector))).*exp.(1im*k_p*x_vector)
#const E_p =  ((F_p_max - 1.1*F_p_sonic)*sech.(x_vector/σ_sech) .+ 1.1*F_p_sonic*heaviside(collect(x_pump_end .- x_vector))).*exp.(1im*k_p*x_vector)

lines(abs.(E_p))

#Size of the box near the horizon in which the correlations will be computed
const ℓ_correlation_real_d = 115
const ℓ_correlation_real_u = 115
const n_correlation_real_start = floor(Int, (x_defect - ℓ_correlation_real_u)/δx)
const n_correlation_real_end = floor(Int, (x_defect + ℓ_correlation_real_d)/δx)


#Definition of FFT quantities
const normalization_ftt = sqrt(x_final)/δx
const matrix_fft = plan_fft(zeros(ComplexF64, n_x), flags=FFTW.ESTIMATE, timelimit=Inf)
const matrix_ifft = plan_ifft(zeros(ComplexF64, n_x), flags=FFTW.ESTIMATE, timelimit=Inf)
const matrix_fft2 = plan_fft(zeros(ComplexF64, (n_x,n_x)), flags=FFTW.ESTIMATE, timelimit=Inf)


#Term of detuning and losses in GP equation
const losses_and_effective_detuning = exp.(-1im*δt*(ω_lp_vector .- ω_p .- 1im*γ_lp/2))


#Function taking into account the non-linearity of the fluid
function PotentialInteractionsAction!(array::AbstractArray{Complex{Float64},1})
    @inbounds for i=1:n_x
        array[i] *= exp(-1im*δt*(potential_lp[i] + g_lp*(abs2(array[i]) - 1/δx)))
    end
end


#Function taking into account the pump and the fluctuations effets
const NoisePumpVector = -1im*δt*E_p
const diffusion_FokkerPlanck = 0#0.5*sqrt.(δt*(γ_lp .+ 2*abs.(imag.(potential_lp)))/δx)
function NoisePumpAction!(array::AbstractArray{Complex{Float64},1})
    @inbounds for i=1:n_x
        array[i] += NoisePumpVector[i] + diffusion_FokkerPlanck[i]*randn(ComplexF64)
    end
end


#Function using the different terms in GP equation to simulate a complete evolution
function OneEvolution!(ϕ_lp::AbstractArray{Complex{Float64},1}, ϕ_lp_fft::AbstractArray{Complex{Float64},1})
    mul!(ϕ_lp_fft, matrix_fft, ϕ_lp)
    ϕ_lp_fft .*= losses_and_effective_detuning
    mul!(ϕ_lp, matrix_ifft, ϕ_lp_fft)
    PotentialInteractionsAction!(ϕ_lp)
    NoisePumpAction!(ϕ_lp)
end


#Function computing the current mean of realizatons at step m for the quantity fourth_order_matrix_real
function RealMatrixFourthOrderMean!(matrix::AbstractArray{Float64,2}, vector::AbstractArray{Float64,1}, m::Int)
    @inbounds for i = n_correlation_real_start:n_correlation_real_end
        @inbounds for j = i:n_correlation_real_end
            matrix[i,j] = (m*matrix[i,j] + vector[i]*vector[j])/(m+1)
        end
    end
end


#Function computing the current mean of realizatons at step m for the quantity second_order_matrix_real
function RealMatrixSecondOrderMean!(matrix::AbstractArray{Float64,2}, vector::AbstractArray{Float64,1}, m::Int)
    @inbounds for i = n_correlation_real_start:n_correlation_real_end
        @inbounds for j = i:n_correlation_real_end
            matrix[i,j] = (m*matrix[i,j] + vector[i] + vector[j])/(m+1)
        end
    end
end


#Function computing the current mean of realizatons at step m of a vector
function VectorMean!(vector1, vector2, m::Int)
    @inbounds for i = 1:n_x
        vector1[i] = (m*vector1[i] + vector2[i])/(m+1)
    end
end


#Function filling the bottom left part of a matrix with the top right part (symmetric matrix)
function FillOtherHalfMatrix!(matrix::AbstractArray{Float64,2})
    @inbounds for j = n_correlation_real_start:n_correlation_real_end
        @inbounds for i = (j+1):(n_correlation_real_end+1)
            matrix[i,j] = matrix[j,i]
        end
    end
    return matrix
end


#Function simulating the evolution of the system until a number of realizations wished
function TWA()
    ϕ_lp = zeros(Complex{Float64}, n_x)
    ϕ_lp_fft = zeros(Complex{Float64}, n_x)
    ϕ_lp_norm = zeros(Float64, n_x)
    ϕ_x = zeros(Complex{Float64}, n_x)
    ϕ_x_norm = zeros(Float64, n_x)
    ϕ_k_norm = zeros(Float64, n_x)
    fourth_order_matrix_real = zeros(Float64, (n_x,n_x))
    second_order_matrix_real = zeros(Float64, (n_x,n_x))

    number_realization = 0

    # for nn in ProgressBar(1:n_time)
    for nn = 1:n_time
        OneEvolution!(ϕ_lp, ϕ_lp_fft)

        time = (nn-1)*δt
        if time == T_ss + number_realization*δt_sample
            VectorMean!(ϕ_x, ϕ_lp, number_realization)
            ϕ_lp_norm .= abs2.(ϕ_lp)
            VectorMean!(ϕ_x_norm, ϕ_lp_norm, number_realization)

            mul!(ϕ_lp_fft, matrix_fft, ϕ_lp)
            VectorMean!(ϕ_k_norm, abs2.(ϕ_lp_fft), number_realization)

            RealMatrixFourthOrderMean!(fourth_order_matrix_real, ϕ_lp_norm, number_realization)
            RealMatrixSecondOrderMean!(second_order_matrix_real, ϕ_lp_norm, number_realization)

            # @printf("Number of realizations: %i, Thread: %i \n", number_realization, Threads.threadid())
            number_realization += 1
        end
    end
    return ϕ_x, ϕ_x_norm, ϕ_k_norm, FillOtherHalfMatrix!(fourth_order_matrix_real), FillOtherHalfMatrix!(second_order_matrix_real)
end


#Function asking to each threads the simulate the evolution of the system until a number of realizations wished
const n_threads = Threads.nthreads()
function MultiThreadsSimulation()
    ϕ_x = zeros(Complex{Float64}, n_x, n_threads)
    ϕ_x_norm = zeros(Float64, n_x, n_threads)
    ϕ_k_norm = zeros(Float64, n_x, n_threads)
    fourth_order_matrix_real = zeros(Float64, n_x, n_x, n_threads)
    second_order_matrix_real = zeros(Float64, n_x, n_x, n_threads)

    @printf("Start of simulation: %s \n", Dates.format(now(), "dd/mm/yyyy at HH:MM:SS"))
    Threads.@threads for thread_number = 1:n_threads
        ϕ_x[:, thread_number],
        ϕ_x_norm[:, thread_number],
        ϕ_k_norm[:, thread_number],
        fourth_order_matrix_real[:, :, thread_number],
        second_order_matrix_real[:, :, thread_number] = TWA()
    end
    @printf("End of simulation: %s \n", Dates.format(now(), "dd/mm/yyyy at HH:MM:SS"))

    ϕ_x = dropdims(mean(ϕ_x, dims=2), dims=2)
    ϕ_x_norm = dropdims(mean(ϕ_x_norm, dims=2), dims=2)
    ϕ_k_norm = dropdims(mean(ϕ_k_norm, dims=2), dims=2)
    fourth_order_matrix_real = dropdims(mean(fourth_order_matrix_real, dims=3), dims=3)
    second_order_matrix_real = dropdims(mean(second_order_matrix_real, dims=3), dims=3)

    return ϕ_x, ϕ_x_norm, ϕ_k_norm, fourth_order_matrix_real, second_order_matrix_real
end


#Function computing the observables with the simulated data
function Observables(data)
    density_x = data[2] .- 1/(2*δx)
    occupation_number_k = data[3] .- 1/2

    ϕ = data[1]
    ϕ_norm = abs2.(ϕ)
    v_x = ħ/m_lp * diff(unwrap(angle.(ϕ[2:n_x]))) / δx
    c_x = sqrt.(ħ*g_lp*abs.(density_x)/m_lp)

    G2_x = data[4] .- (ones((n_x,n_x)) + I)/(2*δx).*(data[5] .- 1/(2*δx))
    normalization_g2_x = density_x*transpose(density_x)
    g2_x = G2_x./normalization_g2_x .- 1

    return density_x, v_x, c_x, g2_x#, occupation_number_
end


#Function writing the observables in text files for export
function WriteFile(observables, folder)
    open(folder*"density.txt", "w") do io
        writedlm(io, observables[1])
    end
    # open(folder*"occupation_number.txt", "w") do io
    #     writedlm(io, observables[6])
    # end
    open(folder*"velocities.txt", "w") do io
        writedlm(io, observables[2])
        writedlm(io, observables[3])
    end
    open(folder*"correlations_real.txt", "w") do io
        @inbounds for i = 1:n_x
            writedlm(io, observables[4][i,:])
        end
    end
end



simulation = MultiThreadsSimulation()
observables = Observables(simulation)
WriteFile(observables, "")
