using KernelAbstractions, FFTW, Logging, Dates, LinearAlgebra, ProgressMeter

choose(x1, x2, m) = isone(m) ? x1 : x2

function first_order_correlations!(dest, sol1::NTuple{1}, sol2::NTuple{1})
    @kernel function kernel!(dest, sol1, sol2)
        a, b, m, n = @index(Global, NTuple)
        idx1 = choose(a, b, m)
        idx2 = choose(a, b, n)
        field1 = choose(sol1, sol2, m)
        field2 = choose(sol1, sol2, n)

        x = zero(eltype(dest))
        for r ∈ axes(sol1, 2)
            x += field1[idx1, r] * conj(field2[idx2, r])
        end
        dest[a, b, m, n] = x / size(sol1, 2)
    end

    backend = get_backend(dest)
    kernel!(backend)(dest, sol1[1], sol2[1], ndrange=size(dest))
end

function second_order_correlations!(dest, sol1::NTuple{1}, sol2::NTuple{1})
    @kernel function kernel!(dest, sol1, sol2)
        j, k = @index(Global, NTuple)
        x = zero(eltype(dest))
        for m ∈ axes(sol1, 2)
            x += abs2(sol1[j, m]) * abs2(sol2[k, m])
        end
        dest[j, k] = x / size(sol1, 2)
    end

    backend = get_backend(dest)
    kernel!(backend)(dest, sol1[1], sol2[1], ndrange=size(dest))
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

function update_correlations!(first_order_r, second_order_r, first_order_k, second_order_k, n_ave, steady_state, kernel1, kernel2,
    lengths, batchsize, nbatches, tspan, dt;
    show_progress=true, noise_eltype=eltype(first(steady_state)), log_path="log.txt", max_datetime=typemax(DateTime),
    rng=Random.default_rng(), kwargs...)
    u0 = map(steady_state) do x
        stack(x for _ ∈ 1:batchsize)
    end

    noise_prototype = similar.(u0, noise_eltype)

    prob = GrossPitaevskiiProblem(u0, lengths; noise_prototype, param, kwargs...)
    solver = StrangSplittingC()

    buffer_first_order = similar(first_order_r)
    buffer_second_order = similar(second_order_r)
    ft_sol1 = similar.(u0)
    ft_sol2 = similar.(u0)

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

        first_order_correlations!(buffer_first_order, sol, sol)
        merge_averages!(first_order_r, n_ave, buffer_first_order, batchsize)
        second_order_correlations!(buffer_second_order, sol, sol)
        merge_averages!(second_order_r, n_ave, buffer_second_order, batchsize)

        for (dest1, dest2, k1, k2, src) ∈ zip(ft_sol1, ft_sol2, kernel1, kernel2, sol)
            mul!(dest1, k1, src)
            mul!(dest2, k2, src)
        end

        first_order_correlations!(buffer_first_order, ft_sol1, ft_sol2)
        merge_averages!(first_order_k, n_ave, buffer_first_order, batchsize)
        second_order_correlations!(buffer_second_order, ft_sol1, ft_sol2)
        merge_averages!(second_order_k, n_ave, buffer_second_order, batchsize)

        n_ave += batchsize
    end

    close(io)
    if !isnothing(progress)
        finish!(progress)
    end

    first_order_r, second_order_r, first_order_k, second_order_k, n_ave
end