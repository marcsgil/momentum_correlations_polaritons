using GeneralizedGrossPitaevskii, CUDA, CUDA.CUFFT, KernelAbstractions, StaticArrays
include("../io.jl")
include("equations.jl")
include("../correlation_kernels.jl")

saving_path = "PositiveP/correlations.h5"
group_name = "no_support"

param, steady_state, t_steady_state, one_point_r, two_point_r, one_point_k, two_point_k, n_ave = h5open(saving_path) do file
    group = file[group_name]

    read_parameters(group),
    reinterpret(reshape, SVector{2,ComplexF32}, group["steady_state"] |> read) |> cu,
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
    one_point_r, two_point_r, one_point_k, two_point_k, n_ave, steady_state, (param.L,), 10^5, 10^3, tspan, param.δt;
    dispersion, potential, nonlinearity, pump, param, noise_func, show_progress=false, noise_eltype=real(eltype(steady_state)),
    max_datetime=DateTime(2025, 2, 27, 8, 0, 0, 0))
##
h5open(saving_path, "cw") do file
    group = file[group_name]
    group["one_point_r"][:] = Array(one_point_r)
    group["two_point_r"][:, :] = Array(two_point_r)
    group["one_point_k"][:] = Array(one_point_k)
    group["two_point_k"][:, :] = Array(two_point_k)
    group["n_ave"][:] = [n_ave]
end