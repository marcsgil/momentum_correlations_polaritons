using KernelAbstractions, FFTW, Logging, Dates, LinearAlgebra

function first_order_correlations!(dest, sol)
    @kernel function kernel!(dest, sol)
        k, k′, m, n = @index(Global, NTuple)
        x = zero(eltype(dest))
        for r ∈ axes(sol, 3)
            x += sol[(k, k′)[m], m, r] * conj(sol[(k, k′)[n], n, r])
        end
        dest[k, k′, m, n] = x / size(sol, 3)
    end

    backend = get_backend(dest)
    kernel!(backend)(dest, sol, ndrange=size(dest))
end

function second_order_correlations!(dest, sol)
    @kernel function kernel!(dest, sol)
        j, k = @index(Global, NTuple)
        x = zero(eltype(dest))
        for m ∈ axes(sol, 3)
            x += abs2(sol[j, 1, m]) * abs2(sol[k, 2, m])
        end
        dest[j, k] = x / size(sol, 3)
    end

    backend = get_backend(dest)
    kernel!(backend)(dest, sol, ndrange=size(dest))
end

function merge_averages!(dest, n_dest, new, n_new)
    @. dest = dest / (1 + n_new / n_dest) + new / (1 + n_dest / n_new)
end

function get_ft_sol(sol::AbstractArray{T}) where {T<:Number}
    ifftshift(fft(fftshift(sol, 1), 1), 1)
end

function update_correlations!(first_order_r, second_order_r, first_order_k, second_order_k, n_ave, steady_state, windows,
    lengths, batchsize, nbatches, tspan, δt;
    show_progress=true, noise_eltype=eltype(steady_state), log_path="log.txt", max_datetime=typemax(DateTime), kwargs...)
    u0 = stack(steady_state for _ ∈ 1:batchsize)
    noise_prototype = similar(u0, noise_eltype)

    prob = GrossPitaevskiiProblem(u0, lengths; noise_prototype, param, kwargs...)
    solver = StrangSplittingC(1, δt)

    buffer_first_order = similar(first_order_r)
    buffer_second_order = similar(second_order_r)

    io = open(log_path, "w+")
    logger = SimpleLogger(io)

    for batch ∈ 1:nbatches
        now() > max_datetime && break
        with_logger(logger) do
            @info "Batch $batch"
        end
        flush(io)

        ts, _sol = GeneralizedGrossPitaevskii.solve(prob, solver, tspan; save_start=false, show_progress)
        _sol = dropdims(_sol, dims=3)
        sol = stack((_sol, _sol), dims=2)
        first_order_correlations!(buffer_first_order, sol)
        merge_averages!(first_order_r, n_ave, buffer_first_order, batchsize)
        second_order_correlations!(buffer_second_order, sol)
        merge_averages!(second_order_r, n_ave, buffer_second_order, batchsize)

        ft_sol = get_ft_sol(sol .* windows)
        first_order_correlations!(buffer_first_order, ft_sol)
        merge_averages!(first_order_k, n_ave, buffer_first_order, batchsize)
        second_order_correlations!(buffer_second_order, ft_sol)
        merge_averages!(second_order_k, n_ave, buffer_second_order, batchsize)

        n_ave += batchsize
    end

    close(io)

    first_order_r, second_order_r, first_order_k, second_order_k, n_ave
end