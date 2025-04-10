using KernelAbstractions, FFTW, Logging, Dates, LinearAlgebra, ProgressMeter, Statistics

merge_averages(μ, n, new_sum, new_n) = μ / (1 + new_n / n) + new_sum / (n + new_n)

function merge_averages!(μ1, n1, μ2, n2)
    @. μ1 = μ1 / (1 + n2 / n1) + μ2 / (1 + n1 / n2)
end

function kahan_step(x, c, next)
    # Apply Kahan summation algorithm
    y = next - c    # Corrected value (value to be added minus compensation)
    t = x + y          # Raw sum (might lose low-order bits)
    c = (t - x) - y    # Calculate new compensation term
    x = t
    x, c
end

@kernel function mean_kernel!(dest, src, f, n_ave)
    j = @index(Global)
    x = zero(eltype(dest))
    c = zero(eltype(dest))  # Compensation term for Kahan summation

    for m ∈ axes(src, 2)
        next = f(src[j, m])
        x, c = kahan_step(x, c, next)  # Apply Kahan summation
    end

    dest[j] = merge_averages(dest[j], n_ave, x, size(src, 2))
end

@kernel function twoD_kernel!(dest1, dest2, src1, src2, n_ave)
    j, k = @index(Global, NTuple)
    x1 = zero(eltype(dest1))
    c1 = zero(eltype(dest1))
    x2 = zero(eltype(dest2))
    c2 = zero(eltype(dest2))

    for m ∈ axes(src1, 2)
        next1 = src1[j, m] * conj(src2[k, m])
        next2 = abs2(next1)
        x1, c1 = kahan_step(x1, c1, next1)  # Apply Kahan summation
        x2, c2 = kahan_step(x2, c2, next2)  # Apply Kahan summation
    end

    dest1[j, k] = merge_averages(dest1[j, k], n_ave, x1, size(src1, 2))
    dest2[j, k] = merge_averages(dest2[j, k], n_ave, x2, size(src1, 2))
end

function update_averages!(averages, sol1, sol2, n_ave)
    backend = get_backend(averages[1])
    mean_kernel!(backend)(averages[1], sol1[1], abs2, n_ave; ndrange=size(averages[1]))
    mean_kernel!(backend)(averages[2], sol2[1], abs2, n_ave; ndrange=size(averages[2]))
    #= μ1 = mean(abs2, sol1[1], dims=2)
    μ2 = mean(abs2, sol2[1], dims=2)
    N = size(sol1[1], 2)
    merge_averages!(averages[1], n_ave, μ1, N)
    merge_averages!(averages[2], n_ave, μ2, N) =#
    twoD_kernel!(backend)(averages[3], averages[4], sol1[1], sol2[1], n_ave; ndrange=size(averages[3]))
end

function windowed_ft!(dest, src, window_func, first_idx, plan)
    N = length(window_func)
    dest .= view(src, first_idx:first_idx+N-1, :) .* window_func
    plan * dest
end

function windowed_ft!(dest, src, window::Window, plan)
    windowed_ft!(dest, src, window.window, window.first_idx, plan)
end

function get_ft_buffers(window_pair, batchsize, steady_state)
    window1 = complex(window_pair.first.window)
    window2 = complex(window_pair.second.window)

    ft_sol1 = map(steady_state) do x
        stack(window1 for _ ∈ 1:batchsize)
    end
    ft_sol2 = map(steady_state) do x
        stack(window2 for _ ∈ 1:batchsize)
    end

    plan1 = plan_fft!(ft_sol1[1], 1)
    plan2 = plan_fft!(ft_sol2[1], 1)

    ft_sol1, ft_sol2, plan1, plan2
end

function update_correlations!(position_averages, momentum_averages, n_ave, steady_state, param, window_pairs, batchsize, nbatches, tspan;
    show_progress=true, noise_eltype=eltype(first(steady_state)), log_path="log.txt", max_datetime=typemax(DateTime),
    rng=nothing, kwargs...)

    u0 = map(steady_state) do x
        stack(x for _ ∈ 1:batchsize)
    end

    noise_prototype = similar.(u0, noise_eltype)

    prob = GrossPitaevskiiProblem(u0, (param.L,); noise_prototype, param, kwargs...)
    solver = StrangSplitting()

    ft_buffers = [get_ft_buffers(pair, batchsize, steady_state) for pair ∈ window_pairs]

    io = open(log_path, "w+")
    logger = SimpleLogger(io)

    steps_per_save = GeneralizedGrossPitaevskii.resolve_fixed_timestepping(param.dt, tspan, 1)[2]
    if show_progress
        progress = Progress(steps_per_save * nbatches)
    else
        progress = nothing
    end

    for batch ∈ 1:nbatches
        now() > max_datetime && break
        with_logger(logger) do
            @info "Batch $batch"
        end
        flush(io)

        sol = map(GeneralizedGrossPitaevskii.solve(prob, solver, tspan; nsaves=1, param.dt, save_start=false, show_progress, progress, rng)[2]) do x
            dropdims(x, dims=3)
        end

        update_averages!(position_averages, sol, sol, n_ave)

        for (averages, ft_buffer, window_pair) ∈ zip(momentum_averages, ft_buffers, window_pairs)
            ft_sol1, ft_sol2, plan1, plan2 = ft_buffer

            for (dest1, dest2, src) ∈ zip(ft_sol1, ft_sol2, sol)
                windowed_ft!(dest1, src, window_pair.first, plan1)
                windowed_ft!(dest2, src, window_pair.second, plan2)
            end

            update_averages!(averages, ft_sol1, ft_sol2, n_ave)
        end

        n_ave += batchsize
    end

    close(io)
    if !isnothing(progress)
        finish!(progress)
    end

    position_averages, momentum_averages, n_ave
end

function update_correlations!(saving_dir, batchsize, nbatches, t_sim; array_type::Type{T}=Array, kwargs...) where {T}
    @assert !isfile(joinpath(saving_dir, "previous_averages.jld2")) "Previous averages file already exists. Please remove it before running the simulation."

    steady_state, param, t_steady_state = read_steady_state(saving_dir, T)
    window_pairs = read_window_pairs(saving_dir, T)
    init_averages(saving_dir, steady_state, t_sim)

    position_averages, momentum_averages, n_ave = read_averages(saving_dir, T)

    tspan = (t_steady_state, t_steady_state + t_sim)

    position_averages, momentum_averages, n_ave = update_correlations!(
        position_averages, momentum_averages, n_ave, steady_state, param, window_pairs, batchsize, nbatches, tspan; param, kwargs...)

    save_averages(saving_dir, position_averages, momentum_averages, n_ave)
end