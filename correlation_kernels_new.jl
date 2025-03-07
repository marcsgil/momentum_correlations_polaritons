using KernelAbstractions, FFTW, Logging, Dates, LinearAlgebra

choose(x1, x2, m) = isone(m) ? x1 : x2

function first_order_correlations!(dest, sol1, sol2)
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
    kernel!(backend)(dest, sol1, sol2, ndrange=size(dest))
end

function second_order_correlations!(dest, sol1, sol2)
    @kernel function kernel!(dest, sol1, sol2)
        j, k = @index(Global, NTuple)
        x = zero(eltype(dest))
        for m ∈ axes(sol1, 2)
            x += abs2(sol1[j, m]) * abs2(sol2[k, m])
        end
        dest[j, k] = x / size(sol1, 2)
    end

    backend = get_backend(dest)
    kernel!(backend)(dest, sol1, sol2, ndrange=size(dest))
end

function merge_averages!(dest, n_dest, new, n_new)
    @. dest = dest / (1 + n_new / n_dest) + new / (1 + n_dest / n_new)
end

function get_ft_sol(sol::AbstractArray{T}) where {T<:Number}
    ifftshift(fft(fftshift(sol, 1), 1), 1)
end

function update_correlations!(first_order_r, second_order_r, first_order_k, second_order_k, n_ave, steady_state, kernel1, kernel2,
    lengths, batchsize, nbatches, tspan, δt;
    show_progress=true, noise_eltype=eltype(steady_state), log_path="log.txt", max_datetime=typemax(DateTime), kwargs...)
    u0 = stack(steady_state for _ ∈ 1:batchsize)
    noise_prototype = similar(u0, noise_eltype)

    prob = GrossPitaevskiiProblem(u0, lengths; noise_prototype, param, kwargs...)
    solver = StrangSplittingC(1, δt)

    buffer_first_order = similar(first_order_r)
    buffer_second_order = similar(second_order_r)
    ft_sol1 = similar(u0)
    ft_sol2 = similar(u0)

    io = open(log_path, "w+")
    logger = SimpleLogger(io)

    for batch ∈ 1:nbatches
        now() > max_datetime && break
        with_logger(logger) do
            @info "Batch $batch"
        end
        flush(io)

        sol = dropdims(GeneralizedGrossPitaevskii.solve(prob, solver, tspan; save_start=false, show_progress)[2], dims=3)

        first_order_correlations!(buffer_first_order, sol, sol)
        merge_averages!(first_order_r, n_ave, buffer_first_order, batchsize)
        second_order_correlations!(buffer_second_order, sol, sol)
        merge_averages!(second_order_r, n_ave, buffer_second_order, batchsize)

        mul!(ft_sol1, kernel1, sol)
        mul!(ft_sol2, kernel2, sol)

        first_order_correlations!(buffer_first_order, ft_sol1, ft_sol2)
        merge_averages!(first_order_k, n_ave, buffer_first_order, batchsize)
        second_order_correlations!(buffer_second_order, ft_sol1, ft_sol2)
        merge_averages!(second_order_k, n_ave, buffer_second_order, batchsize)

        n_ave += batchsize
    end

    close(io)

    first_order_r, second_order_r, first_order_k, second_order_k, n_ave
end