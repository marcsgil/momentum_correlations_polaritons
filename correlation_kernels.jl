using KernelAbstractions, CUDA.CUFFT, Logging, Dates

function one_point_corr!(dest, sol::AbstractArray{T}) where {T<:Number}
    backend = get_backend(dest)

    @kernel function kernel!(dest, sol)
        j = @index(Global)
        x = zero(eltype(dest))
        for k ∈ axes(sol, 2)
            x += abs2(sol[j, k])
        end
        dest[j] = x / size(sol, 2)
    end

    kernel!(backend, 64)(dest, sol, ndrange=length(dest))
    KernelAbstractions.synchronize(backend)
end

function two_point_corr!(dest, sol::AbstractArray{T}) where {T<:Number}
    backend = get_backend(dest)

    @kernel function kernel!(dest, sol)
        j, k = @index(Global, NTuple)
        x = zero(eltype(dest))
        for m ∈ axes(sol, 2)
            x += abs2(sol[j, m]) * abs2(sol[k, m])
        end
        dest[j, k] = x / size(sol, 2)
    end

    kernel!(backend, 64)(dest, sol, ndrange=size(dest))
    KernelAbstractions.synchronize(backend)
end

function one_point_corr!(dest, sol)
    backend = get_backend(dest)

    @kernel function kernel!(dest, sol)
        j = @index(Global)
        x = 0f0
        for k ∈ axes(sol, 2)
            x += real(prod(sol[j, k]))
        end
        dest[j] = x / size(sol, 2)
    end

    kernel!(backend, 64)(dest, sol, ndrange=length(dest))
    KernelAbstractions.synchronize(backend)
end

function two_point_corr!(dest, sol)
    backend = get_backend(dest)

    @kernel function kernel!(dest, sol)
        j, k = @index(Global, NTuple)
        x = 0f0
        for m ∈ axes(sol, 2)
            x += real(prod(sol[j, m]) * prod(sol[k, m]))
        end
        dest[j, k] = x / size(sol, 2)
    end

    kernel!(backend, 64)(dest, sol, ndrange=size(dest))
    KernelAbstractions.synchronize(backend)
end

function merge_averages!(dest, n_dest, new, n_new)
    @. dest = dest / (1 + n_new / n_dest) + new / (1 + n_dest / n_new)
end

function get_ft_sol(sol::AbstractArray{T}) where {T<:Number}
    fftshift(fft(ifftshift(sol, 1), 1), 1)
end

function get_ft_sol(sol)
    αs = first.(sol)
    βs = last.(sol)

    ft_αs = fftshift(fft(ifftshift(αs, 1), 1), 1)
    ft_βs = fftshift(ifft(ifftshift(βs, 1), 1), 1)

    map((α, β) -> SVector(α, β), ft_αs, ft_βs)
end

function update_correlations!(one_point_r, two_point_r, one_point_k, two_point_k, n_ave, steady_state, lengths, batchsize, nbatches, tspan, δt;
    show_progress=true, noise_eltype=eltype(steady_state), log_path="log.txt", max_datetime=typemax(DateTime), kwargs...)
    u0 = stack(steady_state for _ ∈ 1:batchsize)
    noise_prototype = similar(u0, noise_eltype)

    prob = GrossPitaevskiiProblem(u0, lengths; noise_prototype, param, kwargs...)
    solver = StrangSplittingC(1, δt)

    buffer_one_point = similar(one_point_r)
    buffer_two_point = similar(two_point_r)

    io = open(log_path, "w+")
    logger = SimpleLogger(io)

    for batch ∈ 1:nbatches
        now() > max_datetime && break
        with_logger(logger) do
            @info "Batch $batch"
        end
        flush(io)

        ts, _sol = GeneralizedGrossPitaevskii.solve(prob, solver, tspan; save_start=false, show_progress)
        sol = dropdims(_sol, dims=3)
        one_point_corr!(buffer_one_point, sol)
        merge_averages!(one_point_r, n_ave, buffer_one_point, batchsize)
        two_point_corr!(buffer_two_point, sol)
        merge_averages!(two_point_r, n_ave, buffer_two_point, batchsize)

        ft_sol = get_ft_sol(sol)
        one_point_corr!(buffer_one_point, ft_sol)
        merge_averages!(one_point_k, n_ave, buffer_one_point, batchsize)
        two_point_corr!(buffer_two_point, ft_sol)
        merge_averages!(two_point_k, n_ave, buffer_two_point, batchsize)

        n_ave += batchsize
    end

    close(io)

    one_point_r, two_point_r, one_point_k, two_point_k, n_ave
end