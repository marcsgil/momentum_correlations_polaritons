using GeneralizedGrossPitaevskii, CUDA
include("../io.jl")
include("equations.jl")
include("../correlation_kernels.jl")

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
    one_point_r, two_point_r, one_point_k, two_point_k, n_ave, steady_state, (param.L,), 10^4, 2, tspan, param.δt;
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