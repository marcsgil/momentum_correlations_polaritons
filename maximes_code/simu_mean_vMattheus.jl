#flags to control the mean field (Gross-Pitaevskii) trajectory calculation
do_mean = true
import_mean = !do_mean
write_mean = do_mean
plot_mean = true


#flags to control the bogoliubov diagonalization
do_bogo = true
import_bogo = !do_bogo
write_bogo = do_bogo
plot_bogo_eigvals = true

#flags to control additio
plot_dispersion = true
save_dispersion = true
plot_bistability = true
plot_FFTmean = false

#close all previous plots
close("all")


# Function for dynamic plotting of ϕ during calculation
function dynamic_phi_plot(phi::Array{Complex{Float64}}, step::Float64; save::Bool=true)
    figure()
    plot(real.(phi), "b")
    xlabel("x [μm]")
    ylabel("Re(φ)")
    title("Dynamic plot : step $step")
    legend()
    if save
        savefig(folder * "phi_dynamic_step_$step.png")
    end
end


function fill_bogoliubov_matrix(bogo::SparseMatrixCSC{Complex{Float64}}, phi::Array{Complex{Float64}})

    n_size = length(phi)
    for i in 0:n_size-1
        diag_u = (ħ / (2 * m_lp)) * 2 / δx^2 + ω_lp_0 - ω_p + 2 * g_lp * abs2(phi[i+1]) + real(potential_lp[i+1]) - im * γ_lp / 2
        diag_v = -((ħ / (2 * m_lp)) * 2 / δx^2 + ω_lp_0 - ω_p + 2 * g_lp * abs2(phi[i+1]) + real(potential_lp[i+1])) - im * γ_lp / 2
        offdiag = g_lp * phi[i+1]

        if i == 0
            bogo[1, 2*n_size-1] = -(ħ / (2 * m_lp)) * 1 / δx^2
            bogo[2, 2*n_size] = (ħ / (2 * m_lp)) * 1 / δx^2
        else
            bogo[2i+1, 2i-1] = -(ħ / (2 * m_lp)) * 1 / δx^2
            bogo[2i+2, 2i] = (ħ / (2 * m_lp)) * 1 / δx^2
        end

        bogo[2i+1, 2i+1] = diag_u
        bogo[2i+1, 2i+2] = offdiag
        bogo[2i+2, 2i+1] = -conj(offdiag)
        bogo[2i+2, 2i+2] = diag_v


        if i == n_size - 1
            bogo[2i+1, 1] = -(ħ / (2 * m_lp)) * 1 / δx^2
            bogo[2i+2, 2] = (ħ / (2 * m_lp)) * 1 / δx^2
        else
            bogo[2i+1, 2i+3] = -(ħ / (2 * m_lp)) * 1 / δx^2
            bogo[2i+2, 2i+4] = (ħ / (2 * m_lp)) * 1 / δx^2
        end



    end



end

function reorder_eigen(eigvals::Vector{Complex{Float64}}, eigvecs::Matrix{Complex{Float64}})
    perm = sortperm(eigvals, by=x -> real(x))
    return eigvals[perm], eigvecs[:, perm]
end

# Bogoliubov Solver
function solve_bogoliubov(phi::Array{Complex{Float64}})
    bogo = spzeros(Complex{Float64}, 2 * n_x, 2 * n_x)
    fill_bogoliubov_matrix(bogo, phi)

    # Check matrix values
    println("Matrix contains NaN: ", any(isnan, Matrix(bogo)))
    println("Matrix contains Inf: ", any(isinf, Matrix(bogo)))

    # Solve eigenvalue problem
    eig_values, eig_vectors = eigen(Matrix(bogo))
    eig_values, eig_vectors = reorder_eigen(eig_values, eig_vectors)

    # Normalize modes
    for i in 1:size(eig_vectors, 2)
        norm = sqrt(sum(abs2, eig_vectors[:, i]))
        eig_vectors[:, i] ./= norm
    end

    # Print eigenvalues
    # println("Eigenvalues:", eig_values)

    # Output stability
    unstable = any(imag.(eig_values) .> -γ_lp / 2)
    println("System is ", if unstable
        "unstable"
    else
        "stable"
    end)

    return eig_values, eig_vectors
end


#Function taking into account the non-linearity of the fluid
function PotentialInteractionsAction!(array::AbstractArray{Complex{Float64},1})
    @inbounds for i = 1:n_x
        array[i] *= exp(-1im * δt * (potential_lp[i] + g_lp * (abs2(array[i]) - 1 / δx)))
    end
end

function NoisePumpAction!(array::AbstractArray{Complex{Float64},1})
    @inbounds for i = 1:n_x
        array[i] += NoisePumpVector[i] + diffusion_FokkerPlanck[i] * randn(ComplexF64)
    end
end
function PumpAction!(array::AbstractArray{Complex{Float64},1})
    @inbounds for i = 1:n_x
        array[i] += NoisePumpVector[i]
    end
end
function ProbeAction!(array::AbstractArray{Complex{Float64},1}, bogo::Vector{Float64}, E_s::AbstractArray{Complex{Float64},1}, k_s::Float64, time::Float64)
    ω_s = resdisp(0, 1, k_s)
    @inbounds for i = 1:n_x
        array[i] += -1im * δt * E_s[i] * exp(-1im * ω_s * time)
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
function OneEvolutionNoNoise!(ϕ_lp::AbstractArray{Complex{Float64},1}, ϕ_lp_fft::AbstractArray{Complex{Float64},1})
    mul!(ϕ_lp_fft, matrix_fft, ϕ_lp)
    ϕ_lp_fft .*= losses_and_effective_detuning
    mul!(ϕ_lp, matrix_ifft, ϕ_lp_fft)
    PotentialInteractionsAction!(ϕ_lp)
    PumpAction!(ϕ_lp)
end
function OneEvolutionProbe!(ϕ_lp::AbstractArray{Complex{Float64},1}, ϕ_lp_fft::AbstractArray{Complex{Float64},1}, bogo::Vector{Float64}, E_s::AbstractArray{Complex{Float64},1}, k_s::Float64, time::Float64)
    mul!(ϕ_lp_fft, matrix_fft, ϕ_lp)
    ϕ_lp_fft .*= losses_and_effective_detuning
    mul!(ϕ_lp, matrix_ifft, ϕ_lp_fft)
    PotentialInteractionsAction!(ϕ_lp)
    PumpAction!(ϕ_lp)
    ProbeAction!(ϕ_lp, bogo, E_s, k_s, time)
end
function OneEvolutionNoiseProbe!(ϕ_lp::AbstractArray{Complex{Float64},1}, ϕ_lp_fft::AbstractArray{Complex{Float64},1}, bogo::Vector{Float64}, E_s::AbstractArray{Complex{Float64},1}, k_s::Float64, time::Float64)
    mul!(ϕ_lp_fft, matrix_fft, ϕ_lp)
    ϕ_lp_fft .*= losses_and_effective_detuning
    mul!(ϕ_lp, matrix_ifft, ϕ_lp_fft)
    PotentialInteractionsAction!(ϕ_lp)
    NoisePumpAction!(ϕ_lp)
    ProbeAction!(ϕ_lp, bogo, E_s, k_s, time)
end


#Function computing the current mean of realizatons at step m for the quantity fourth_order_matrix_real
function RealMatrixFourthOrderMean!(matrix::AbstractArray{Float64,2}, vector::AbstractArray{Float64,1}, m::Int)
    @inbounds for i = n_correlation_real_start:n_correlation_real_end
        @inbounds for j = i:n_correlation_real_end
            matrix[i, j] = (m * matrix[i, j] + vector[i] * vector[j]) / (m + 1)
        end
    end
end

#Function computing the current mean of realizatons at step m for the quantity second_order_matrix_real
function RealMatrixSecondOrderMean!(matrix::AbstractArray{Float64,2}, vector::AbstractArray{Float64,1}, m::Int)
    @inbounds for i = n_correlation_real_start:n_correlation_real_end
        @inbounds for j = i:n_correlation_real_end
            matrix[i, j] = (m * matrix[i, j] + vector[i] + vector[j]) / (m + 1)
        end
    end
end

#Function computing the current mean of realizatons at step m of a vector
function VectorMean!(vector1, vector2, m::Int)
    @inbounds for i = 1:n_x
        vector1[i] = (m * vector1[i] + vector2[i]) / (m + 1)
    end
end

#Function filling the bottom left part of a matrix with the top right part (symmetric matrix)
function FillOtherHalfMatrix!(matrix::AbstractArray{Float64,2})
    @inbounds for j = n_correlation_real_start:n_correlation_real_end
        @inbounds for i = (j+1):(n_correlation_real_end+1)
            matrix[i, j] = matrix[j, i]
        end
    end
    return matrix
end


#Function simulating the evolution of the system until the stationary state
function SimulationSteadyState(EvolutionFunction)
    ϕ_lp = zeros(Complex{Float64}, n_x)
    ϕ_lp_fft = zeros(Complex{Float64}, n_x)
    time_series = Float64[]              # Collect norms directly into this array

    println("Starting time evolution...")
    for step = 1:n_stationary
        # println("Step $step: Starting computation...") # useful for debugging, be careful when doing long simulations
        EvolutionFunction(ϕ_lp, ϕ_lp_fft)

        # Collect and store norm
        norm_phi = norm(ϕ_lp)
        push!(time_series, norm_phi)  # Append norm to time series array
        # println("Step $step: norm = $norm_phi") #useful for debugging, be careful when doing long simulations

        # Check for NaN values
        if any(isnan, ϕ_lp)
            println("NaN detected at step $step")
            break
        end

        # Output every "EVERY" steps -- this is useful for debugging. Adjust value of "EVERY" at beginning of code to choose at which step to plot.
        # if step % EVERY == 0 || step == n_stationary
        #     println("Step $step: Plotting dynamic state of ϕ_lp...")
        #     # Dynamic plot for the current step
        #     dynamic_phi_plot(ϕ_lp, step)
        # end
    end
    println("Time evolution complete.")

    # Generate and save the final plot
    println("Generating final time series plot...")
    plot(time_series)
    xlabel("Time Step"),
    ylabel("Norm of φ"),
    title("Evolution of φ")
    savefig(folder * "final_time_series_plot.png")
    println("Final plot saved to: ", folder * "final_time_series_plot.png")
    return ϕ_lp
end

###################################################
###################################################
#MEAN PLOTS
if do_mean
    ϕmean_x = zeros(Complex{Float64}, n_x)
    ϕmean_x = SimulationSteadyState(OneEvolutionNoNoise!)
elseif import_mean
    ϕmean_x = zeros(Complex{Float64}, n_x)
    re_ϕmean_x = readdlm(folder * "new_mean_field_re.txt", Float64)
    im_ϕmean_x = readdlm(folder * "new_mean_field_im.txt", Float64)
    for i in eachindex(re_ϕmean_x)
        ϕmean_x[i] = re_ϕmean_x[i] .+ 1im * im_ϕmean_x[i]
    end
end
if write_mean
    open(folder * "new_mean_field_re.txt", "w") do io
        @inbounds for i = 1:n_x
            writedlm(io, real(ϕmean_x[i]))
        end
    end
    open(folder * "new_mean_field_im.txt", "w") do io
        @inbounds for i = 1:n_x
            writedlm(io, imag(ϕmean_x[i]))
        end
    end
end
if plot_mean
    figure()
    plot(x_vector .- x_defect, sqrt.(ħ * g_lp * abs2.(ϕmean_x) / m_lp), "b", label="c_x")
    plot(x_vector[2:n_x-1] .- x_defect, ħ / m_lp * diff(unwrap(angle.(ϕmean_x[2:n_x]))) / δx, "r", label="v_x")
    xlabel("x [μm]")
    ylabel("velocities [μm/ps]")
    ylim([0, 6])
    xlim([x_beg_u - x_defect, x_end_d - x_defect])
    title("Velocities of the meanfield")
    legend()
    savefig(folder * "velocities.png")
    save_vector(sqrt.(ħ * g_lp * abs2.(ϕmean_x) / m_lp), "c_x.txt")
    save_vector(ħ / m_lp * diff(unwrap(angle.(ϕmean_x[2:n_x]))) / δx, "v_x.txt")    
end

###################################################
###################################################
#BOGO PLOTS
if do_bogo
    eig_values = zeros(Complex{Float64}, 2 * n_x)
    eig_vectors = zeros(Complex{Float64}, 2 * n_x, 2 * n_x)
    # Diagonalising the Bogoliubov matrix
    eig_values, eig_vectors = solve_bogoliubov(ϕmean_x)
    println("Bogoliubov matrix diagonalised.")

elseif import_bogo
    eig_values = zeros(Complex{Float64}, 2 * n_x)
    eig_vectors = zeros(Complex{Float64}, 2 * n_x, 2 * n_x)

    eig_values = readdlm("eigenvalues.txt", '\t', Complex{Float64}, '\n')
    eig_vectors = readdlm("eigenvectors.txt", '\t', Complex{Float64}, '\n')
end
if write_bogo
    open(folder * "eigenvalues.txt", "w") do io
        # @inbounds for i = 1:size(eig_values, 1)
        writedlm(io, eig_values)
        # end
    end
    open(folder * "eigenvectors.txt", "w") do io
        # @inbounds for i = 1:size(eig_vectors, 1)
        writedlm(io, eig_vectors)
        # end
    end
end

#Cette fonction n'est pas aboutie, c'était pour calculer la norme des modes
function calculate_norm(vect)
    norm = 0
    # for i in 1:n_x
    norm += sum(abs2.(vect[1:n_x])) - sum(abs2.(vect[n_x+1:2*n_x]))
    # end
    return norm
end

if plot_bogo_eigvals
    figure()
    for j in 1:length(eig_values)
        if calculate_norm(eig_vectors[:, j]) > 0
            plot(ħ * real.(eig_values[j]), imag.(eig_values[j]) / γ_lp, "ro")
        elseif calculate_norm(eig_vectors[:, j]) < 0
            plot(ħ * real.(eig_values[j]), imag.(eig_values[j]) / γ_lp, color="black", "o")
        elseif calculate_norm(eig_vectors[:, j]) == 0
            plot(ħ * real.(eig_values[j]), imag.(eig_values[j]) / γ_lp, color="blue", "o")
        end
    end
    xlabel("ħ Re(ω)")
    ylabel("Im(ω)/γ")
    xlim([0, 1])
    title("Eigenvalues of the Bogoliubov matrix")
    savefig(folder * "eigenvalues.png")
end





###################################################
###################################################
###############Other plots
if plot_dispersion
    const v_x = ħ / m_lp * diff(unwrap(angle.(ϕmean_x[2:n_x]))) / δx
    const c_x = sqrt.(ħ * g_lp * abs2.(ϕmean_x) / m_lp)
    const extraction_upstream_homogeneous = (floor(Int64, (x_beg_u) / δx)+1):1:(floor(Int64, (x_end_u) / δx))
    const extraction_downstream_homogeneous = (floor(Int64, (x_beg_d) / δx)+1):1:(floor(Int64, (x_end_d) / δx))

    # Definition of k_u and k_d
    const density_u = m_lp * (mean(c_x[extraction_upstream_homogeneous]))^2 / (ħ * g_lp)
    const v_u = mean(v_x[extraction_upstream_homogeneous]) # average flow velocity in the upstream region
    const k_u = m_lp * v_u / ħ # wave vector in the upstream region
    const δk_u = k_vect .- k_u # wave-vector in the upstream region in the lab fourth_order_matrix_real
    const detuning_u = ħ * ω_p - ħ * (ω_lp_0 + ħ * k_u^2 / (2 * m_lp))
    const density_d = m_lp * (mean(c_x[extraction_downstream_homogeneous]))^2 / (ħ * g_lp)
    const v_d = mean(v_x[extraction_downstream_homogeneous]) # average flow velocity in the downstream region
    const k_d = m_lp * v_d / ħ # wave vector in the downstream region
    const δk_d = k_vect .- k_d # wave-vector in the downstream region in the lab fourth_order_matrix_real
    const detuning_d = ħ * ω_p - ħ * (ω_lp_0 + ħ * k_d^2 / (2 * m_lp))

    # dispersion relation in the up/downstream regions in the lab frame
    ω_u_pos = v_u .* δk_u + .√(abs.((ħ .* δk_u .^ 2 / (2 * m_lp) .+ g_lp .* density_u .- detuning_u / ħ) .* (ħ .* δk_u .^ 2 / (2 * m_lp) .+ 3 * g_lp .* density_u .- detuning_u / ħ))) # imaginary part -u*γ_lp/2 ignored
    ω_u_neg = v_u .* δk_u - .√(abs.((ħ .* δk_u .^ 2 / (2 * m_lp) .+ g_lp .* density_u .- detuning_u / ħ) .* (ħ .* δk_u .^ 2 / (2 * m_lp) .+ 3 * g_lp .* density_u .- detuning_u / ħ))) # imaginary part -u*γ_lp/2 ignored
    #downstream with pump term
    # ω_d_pos = v_d .* δk_d + .√(abs.((ħ .* δk_d.^2 / (2* m_lp) .+ g_lp .* density_d .- detuning_d/ħ) .* (ħ .* δk_d.^2 / (2 * m_lp) .+ 3 * g_lp .* density_d .- detuning_d/ħ))) # imaginary part -u*γ_lp/2 ignored
    # ω_d_neg = v_d .* δk_d - .√(abs.((ħ .* δk_d.^2 / (2* m_lp) .+ g_lp .* density_d .- detuning_d/ħ) .* (ħ .* δk_d.^2 / (2 * m_lp) .+ 3 * g_lp .* density_d .- detuning_d/ħ))) # imaginary part -u*γ_lp/2 ignored
    #downstream without pump term (ballistic)
    ω_d_pos = v_d .* δk_d + .√(abs.((ħ .* δk_d .^ 2 / (2 * m_lp) .* (ħ .* δk_d .^ 2 / (2 * m_lp) .+ g_lp .* density_d)))) # imaginary part -u*γ_lp/2 ignored
    ω_d_neg = v_d .* δk_d - .√(abs.((ħ .* δk_d .^ 2 / (2 * m_lp) .* (ħ .* δk_d .^ 2 / (2 * m_lp) .+ g_lp .* density_d)))) # imaginary part -u*γ_lp/2 ignored


    figure()
    plot(k_vect, ω_u_pos, "b", label="positiv norm")
    xlim(-1, 2)
    ylim(-1, 1)
    plot(k_vect, ω_u_neg, "r", label="negativ norm")
    title("Upstream dispersion")
    legend()
    savefig(folder * "ana_upstreamdispersion")

    figure()
    plot(k_vect, ω_d_pos, "b")
    xlim(-1, 2)
    ylim(-1, 1)
    plot(k_vect, ω_d_neg, "r")
    title("Downstream dispersion")
    legend()
    savefig(folder * "ana_downstreamdispersion")
end

if save_dispersion
    save_vector(k_vect, "k_vect.txt")
    save_vector(ω_u_pos, "omeg_u_pos.txt")
    save_vector(ω_u_neg, "omeg_u_neg.txt")
    save_vector(ω_d_pos, "omeg_d_pos.txt")
    save_vector(ω_d_neg, "omeg_d_neg.txt")
end

if plot_bistability
    const c_excit_vect = LinRange(0, 2, 10000)
    const F_p_vect_u = sqrt(m_lp / (g_lp * ħ)) .* sqrt.(((m_lp .* c_excit_vect .^ 2 .- detuning_u) ./ ħ) .^ 2 .+ γ_lp^2 / 4) .* c_excit_vect
    const F_p_vect_d = sqrt(m_lp / (g_lp * ħ)) .* sqrt.(((m_lp .* c_excit_vect .^ 2 .- detuning_d) ./ ħ) .^ 2 .+ γ_lp^2 / 4) .* c_excit_vect
    figure(figsize=(4, 6))
    plot(F_p_vect_u, c_excit_vect, "g", label="Upstream")
    plot(F_p_support_u, mean(c_x[extraction_upstream_homogeneous]), "go", label="Operating point upstream")
    plot(F_p_vect_d, c_excit_vect, "c", label="Downstream")
    plot(F_p_support_d, mean(c_x[extraction_downstream_homogeneous]), "co", label="Operating point downstream")
    title("Bistability curve for the flow")
    xlabel("Pumping intensity F_p")
    ylabel("Velocity of sound c")
    xlim([0, 10])
    legend()
    savefig(folder * "bistability")
end

if plot_FFTmean

    nbeg = floor(Int64, 402 / δx)
    nend = floor(Int64, 748 / δx)
    fl_vel = v_x[nbeg:nend]
    # fl_vel = fl_vel .- mean(fl_vel)
    # sound_vel = sound_vel .- mean(sound_vel)

    # fft_fl_vel = fftshift(fft(fl_vel))
    # fft_sound_vel = fftshift(fft(sound_vel))

    fl_vel = (fl_vel .- minimum(fl_vel)) / (maximum(fl_vel) - minimum(fl_vel))
    sound_vel = (abs2.(ϕmean_x[nbeg:nend]) .- minimum(abs2.(ϕmean_x[nbeg:nend]))) / (maximum(abs2.(ϕmean_x[nbeg:nend])) - minimum(abs2.(ϕmean_x[nbeg:nend])))
    fl_vel = fl_vel .- mean(fl_vel)
    sound_vel = sound_vel .- mean(sound_vel)
    fft_fl_vel = fftshift(fft(fl_vel))
    fft_sound_vel = fftshift(fft(sound_vel))

    δkv = 2 * π / (nend - nbeg + 1)
    n_k = length(fl_vel)
    freq = LinRange(-(n_k / 2 - 1) * δkv, (n_k / 2) * δkv, n_k)

    figure()
    title("Fourier transform of the meanfield oscillations downstream")
    subplot(211)
    plot(freq, abs.(fft_sound_vel), label="density")
    xlim([-0.7, 0.7])
    legend()
    subplot(212)
    plot(freq, abs.(fft_fl_vel), label="fluid velocity", "r")
    xlim([-0.7, 0.7])
    legend()
    xlabel("k")
    ylabel("TF")
    print(freq[argmax(abs.(fft_fl_vel))])

    figure()
    title("Oscillations downstream")
    subplot(211)
    plot(x_vector[nbeg:nend], sound_vel, label="density")
    legend()
    subplot(212)
    plot(x_vector[nbeg:nend], fl_vel, label="fluid velocity", "r")
    legend()
    ylabel("density (norm)")
    xlabel("x [μm]")

end
