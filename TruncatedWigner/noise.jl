using GeneralizedGrossPitaevskii, CUDA, CUDA.CUFFT, KernelAbstractions
include("../io.jl")
include("equations.jl")

function one_point_corr!(dest, sol)
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

function two_point_corr!(dest, sol)
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

function merge_averages!(dest, n_dest, new, n_new)
    @. dest = dest / (1 + n_new / n_dest) + new / (1 + n_dest / n_new)
end

function update_correlations!(one_point_r, two_point_r, one_point_k, two_point_k, n_ave, steady_state, lengths, batchsize, nbatches, tspan, δt; show_progress=true, kwargs...)
    u0 = stack(steady_state for _ ∈ 1:batchsize)
    noise_prototype = similar(u0)

    prob = GrossPitaevskiiProblem(u0, lengths; noise_prototype, param, kwargs...)
    solver = StrangSplittingC(1, δt)

    buffer_one_point = similar(one_point_r)
    buffer_two_point = similar(two_point_r)

    for batch ∈ 1:nbatches
        @info "Batch $batch"
        ts, _sol = GeneralizedGrossPitaevskii.solve(prob, solver, tspan; save_start=false, show_progress)
        sol = dropdims(_sol, dims=3)
        one_point_corr!(buffer_one_point, sol)
        merge_averages!(one_point_r, n_ave, buffer_one_point, batchsize)
        two_point_corr!(buffer_two_point, sol)
        merge_averages!(two_point_r, n_ave, buffer_two_point, batchsize)

        ft_sol = fftshift(fft(ifftshift(sol, 1), 1), 1)
        one_point_corr!(buffer_one_point, ft_sol)
        merge_averages!(one_point_k, n_ave, buffer_one_point, batchsize)
        two_point_corr!(buffer_two_point, ft_sol)
        merge_averages!(two_point_k, n_ave, buffer_two_point, batchsize)

        n_ave += batchsize
    end

    one_point_r, two_point_r, one_point_k, two_point_k, n_ave
end
##
saving_path = "TruncatedWigner/test.h5"
group_name = "TruncatedWigner/test"

param, steady_state, t_steady_state, one_point_r, two_point_r, one_point_k, two_point_k, n_ave = h5open(saving_path) do file
    group = file[group_name]

    read_parameters(group),
    group["steady_state"] |> read |> cu,
    group["t_steady_state"] |> read,
    group["one_point_r"] |> read |> cu,
    group["two_point_r"] |> read |> cu,
    group["one_point_k"] |> read |> cu,
    group["two_point_k"] |> read |> cu,
    group["n_ave"][1]
end
##
tspan = (0.0f0, 50.0f0) .+ t_steady_state

one_point_r, two_point_r, one_point_k, two_point_k, n_ave = update_correlations!(
    one_point_r, two_point_r, one_point_k, two_point_k, n_ave, steady_state, (param.L,), 10^5, 2, tspan, param.δt;
    dispersion, potential, nonlinearity, pump, param, noise_func, show_progress=true)
##
h5open(saving_path, "cw") do file
    group = file[group_name]
    group["one_point_r"][:] = Array(one_point_r)
    group["two_point_r"][:, :] = Array(two_point_r)
    group["one_point_k"][:] = Array(one_point_k)
    group["two_point_k"][:, :] = Array(two_point_k)
    group["n_ave"][:] = [n_ave]
end