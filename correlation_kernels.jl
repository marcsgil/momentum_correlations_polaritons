using KernelAbstractions, FFTW, Logging, Dates, LinearAlgebra, ProgressMeter, Statistics

@kernel function mean_kernel!(dest, src, n)
    j = @index(Global)
    μ = dest[j]

    N = n

    for m ∈ axes(src, 2)
        N += 1
        μ += (abs2(src[j, m]) - μ) / N
    end

    dest[j] = μ
end

@kernel function covariance_kernel!(dest, src1, src2, μ1, μ2, n)
    j, k = @index(Global, NTuple)
    σ² = dest[j, k]
    m1 = μ1[j]
    m2 = μ2[k]

    N = n

    for m ∈ axes(src1, 2)
        N += 1
        m1 += (abs2(src1[j, m]) - m1) / N
        σ² += ((abs2(src1[j, m]) - m1) * (abs2(src2[k, m]) - m2) - σ²) / N
        m2 += (abs2(src2[k, m]) - m2) / N
    end

    dest[j, k] = σ²
end

@kernel function mean_prod_kernel!(dest, src1, src2, n)
    j, k = @index(Global, NTuple)
    μ = dest[j, k]

    N = n

    for m ∈ axes(src1, 2)
        N += 1
        μ += (src1[j, m] * conj(src2[k, m]) - μ) / N
    end

    dest[j, k] = μ
end

function update_averages!(averages, sol1, sol2, n_ave)
    backend = get_backend(averages[1])
    covariance_kernel!(backend)(averages[4], sol1[1], sol2[1], averages[1], averages[2], n_ave; ndrange=size(averages[4]))
    mean_kernel!(backend)(averages[1], sol1[1], n_ave; ndrange=size(averages[1]))
    mean_kernel!(backend)(averages[2], sol2[1], n_ave; ndrange=size(averages[2]))
    mean_prod_kernel!(backend)(averages[3], sol1[1], sol2[1], n_ave; ndrange=size(averages[3]))
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

_randn!(::Nothing, args...) = randn!(args...)
_randn!(args...) = randn!(args...)

function update_correlations!(position_averages, momentum_averages, n_ave, steady_state, param, window_pairs, batchsize, nbatches, tspan;
    show_progress=true, noise_eltype=eltype(first(steady_state)), log_path="log.txt", max_datetime=typemax(DateTime),
    rng=nothing, kwargs...)

    u0 = map(steady_state) do x
        stack(x for _ ∈ 1:batchsize)
    end

    for x ∈ steady_state
        _randn!(rng, x)
        x ./= sqrt(2param.dx)
    end

    noise_prototype = similar.(u0, noise_eltype)

    prob = GrossPitaevskiiProblem(u0, (param.L,); noise_prototype, param, kwargs...)
    solver = StrangSplitting()

    ft_buffers = [get_ft_buffers(pair, batchsize, steady_state) for pair ∈ window_pairs]

    io = open(log_path, "w+")
    logger = SimpleLogger(io)

    steps_per_save = GeneralizedGrossPitaevskii.resolve_fixed_timestepping(param.dt, tspan, 1)[3]
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

    #tspan = (t_steady_state, t_steady_state + t_sim)
    tspan = (zero(t_steady_state), t_steady_state)

    position_averages, momentum_averages, n_ave = update_correlations!(
        position_averages, momentum_averages, n_ave, steady_state, param, window_pairs, batchsize, nbatches, tspan; param, kwargs...)

    save_averages(saving_dir, position_averages, momentum_averages, n_ave)
end