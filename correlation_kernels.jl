using KernelAbstractions, FFTW, Logging, Dates, LinearAlgebra, ProgressMeter

choose(x1, x2, m) = isone(m) ? x1 : x2

function kahan_step(x, c, next)
    # Apply Kahan summation algorithm
    y = next - c    # Corrected value (value to be added minus compensation)
    t = x + y          # Raw sum (might lose low-order bits)
    c = (t - x) - y    # Calculate new compensation term
    x = t
    x, c
end

@kernel function mean_prod_kernel!(dest, src1, src2, f1, f2)
    j, k = @index(Global, NTuple)
    x = zero(eltype(dest))
    c = zero(eltype(dest))  # Compensation term for Kahan summation

    for m ∈ axes(src1, 2)
        # Calculate product for this element
        next = f1(src1[j, m]) * f2(src2[k, m])
        x, c = kahan_step(x, c, next)  # Apply Kahan summation
    end

    dest[j, k] = x / size(src1, 2)
end

function first_order_correlations!(dest, sol1::NTuple{1}, sol2::NTuple{1})
    @kernel function kernel!(dest, sol1, sol2)
        a, b, m, n = @index(Global, NTuple)
        idx1 = choose(a, b, m)
        idx2 = choose(a, b, n)
        field1 = choose(sol1, sol2, m)
        field2 = choose(sol1, sol2, n)

        x = zero(eltype(dest))
        c = zero(eltype(dest))  # Compensation term for Kahan summation
        for r ∈ axes(sol1, 2)
            next = field1[idx1, r] * conj(field2[idx2, r])
            x, c = kahan_step(x, c, next)  # Apply Kahan summation
        end 
        dest[a, b, m, n] = x / size(sol1, 2)
    end

    backend = get_backend(dest)
    kernel!(backend)(dest, sol1[1], sol2[1], ndrange=size(dest))
end

function second_order_correlations!(dest, sol1::NTuple{1}, sol2::NTuple{1})
    mean_prod_kernel!(get_backend(dest))(dest, sol1[1], sol2[1], abs2, abs2, ndrange=size(dest))
end

function first_order_correlations!(dest, sol1::NTuple{2}, sol2::NTuple{2})
    @kernel function kernel!(dest, sol1, sol2)
        a, b, m = @index(Global, NTuple)
        idx = choose(a, b, m)
        field = choose(sol1, sol2, m)

        x = zero(eltype(dest))
        for r ∈ axes(sol1[1], 2)
            x += field[1][idx, r] * field[2][idx, r]
        end
        dest[a, b, m] = x / size(first(sol1), 2)
    end

    backend = get_backend(dest)
    kernel!(backend)(dest, sol1, sol2, ndrange=size(dest))
end

function second_order_correlations!(dest, sol1::NTuple{2}, sol2::NTuple{2})
    @kernel function kernel!(dest, sol1, sol2)
        j, k = @index(Global, NTuple)
        x = zero(eltype(dest))
        for m ∈ axes(sol1[1], 2)
            x += sol1[1][j, m] * sol1[2][j, m] * sol2[1][k, m] * sol2[2][k, m]
        end
        dest[j, k] = x / size(sol1[1], 2)
    end

    backend = get_backend(dest)
    kernel!(backend)(dest, sol1, sol2, ndrange=size(dest))
end

function merge_averages!(dest, n_dest, new, n_new)
    @. dest = dest / (1 + n_new / n_dest) + new / (1 + n_dest / n_new)
end

function windowed_ft!(dest, src, window_func, first_idx, plan)
    N = length(window_func)
    dest .= view(src, first_idx:first_idx+N-1, :) .* window_func
    plan * dest
end

function update_correlations!(first_order_r, second_order_r, first_order_k, second_order_k, n_ave, steady_state, window1, window2, first_idx1, first_idx2,
    lengths, batchsize, nbatches, tspan, dt;
    show_progress=true, noise_eltype=eltype(first(steady_state)), log_path="log.txt", max_datetime=typemax(DateTime),
    rng=nothing, kwargs...)
    u0 = map(steady_state) do x
        stack(x for _ ∈ 1:batchsize)
    end

    noise_prototype = similar.(u0, noise_eltype)

    prob = GrossPitaevskiiProblem(u0, lengths; noise_prototype, param, kwargs...)
    solver = StrangSplitting()

    buffer_first_order_r = similar(first_order_r)
    buffer_second_order_r = similar(second_order_r)
    buffer_first_order_k = similar(first_order_k)
    buffer_second_order_k = similar(second_order_k)

    ft_sol1 = map(steady_state) do x
        stack(window1 for _ ∈ 1:batchsize)
    end
    ft_sol2 = map(steady_state) do x
        stack(window2 for _ ∈ 1:batchsize)
    end

    plan1 = plan_fft!(ft_sol1[1], 1)
    plan2 = plan_fft!(ft_sol2[1], 1)

    io = open(log_path, "w+")
    logger = SimpleLogger(io)

    steps_per_save = GeneralizedGrossPitaevskii.resolve_fixed_timestepping(dt, tspan, 1)[2]
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

        sol = map(GeneralizedGrossPitaevskii.solve(prob, solver, tspan; nsaves=1, dt, save_start=false, show_progress, progress, rng)[2]) do x
            dropdims(x, dims=3)
        end

        first_order_correlations!(buffer_first_order_r, sol, sol)
        merge_averages!(first_order_r, n_ave, buffer_first_order_r, batchsize)
        second_order_correlations!(buffer_second_order_r, sol, sol)
        merge_averages!(second_order_r, n_ave, buffer_second_order_r, batchsize)

        for (dest1, dest2, src) ∈ zip(ft_sol1, ft_sol2, sol)
            windowed_ft!(dest1, src, window1, first_idx1, plan1)
            windowed_ft!(dest2, src, window2, first_idx2, plan2)
        end

        first_order_correlations!(buffer_first_order_k, ft_sol1, ft_sol2)
        merge_averages!(first_order_k, n_ave, buffer_first_order_k, batchsize)
        second_order_correlations!(buffer_second_order_k, ft_sol1, ft_sol2)
        merge_averages!(second_order_k, n_ave, buffer_second_order_k, batchsize)

        n_ave += batchsize
    end

    close(io)
    if !isnothing(progress)
        finish!(progress)
    end

    first_order_r, second_order_r, first_order_k, second_order_k, n_ave
end