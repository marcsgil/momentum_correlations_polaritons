using GeneralizedGrossPitaevskii, CUDA
include("../io.jl")
include("equations.jl")
include("../correlation_kernels.jl")

CUDA.device!(0)

saving_path = "/home/stagios/Marcos/LEON_Marcos/Users/Marcos/MomentumCorrelations/PositiveP/correlations.h5"
group_name = "test_new"

param, steady_state, t_steady_state, one_point_r, two_point_r, one_point_k, two_point_k, n_ave, kernel1, kernel2 = h5open(saving_path) do file
    group = file[group_name]

    read_parameters(group),
    (group["steady_state"][:, 1], group["steady_state"][:, 2]) .|> cu,
    group["t_steady_state"] |> read,
    group["one_point_r"] |> read |> cu,
    group["two_point_r"] |> read |> cu,
    group["one_point_k"] |> read |> cu,
    group["two_point_k"] |> read |> cu,
    group["n_ave"][1],
    (group["kernel1"][:, :, 1], group["kernel1"][:, :, 2]) .|> cu,
    (group["kernel2"][:, :, 1], group["kernel2"][:, :, 2]) .|> cu
end
##
tspan = (0.0f0, 50.0f0) .+ t_steady_state

one_point_r, two_point_r, one_point_k, two_point_k, n_ave = update_correlations!(
    one_point_r, two_point_r, one_point_k, two_point_k, n_ave, steady_state, kernel1, kernel2, (param.L,), 10^5, 10^3, tspan, param.δt;
    dispersion, potential, nonlinearity, pump, param, noise_func, show_progress=false, noise_eltype=Float32,
    max_datetime=DateTime(2025, 3, 10, 15, 20));

h5open(saving_path, "cw") do file
    group = file[group_name]
    group["one_point_r"][:, :, :] = Array(one_point_r)
    group["two_point_r"][:, :] = Array(two_point_r)
    group["one_point_k"][:, :, :] = Array(one_point_k)
    group["two_point_k"][:, :] = Array(two_point_k)
    group["n_ave"][:] = [n_ave]
end