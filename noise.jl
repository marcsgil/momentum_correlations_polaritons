using GeneralizedGrossPitaevskii, CUDA, Random
include("io.jl")
include("equations.jl")
include("correlation_kernels.jl")

saving_path = "/Volumes/partages/EQ15B/LEON-15B/Users/Marcos/MomentumCorrelations/correlations.h5"
group_name = "test"

#reset_simulation!(saving_path, group_name)

param, steady_state, t_steady_state, one_point_r, two_point_r, one_point_k, two_point_k, n_ave, window1, window2, first_idx1, first_idx2 = h5open(saving_path) do file
    group = file[group_name]

    read_parameters(group),
    (group["steady_state"] |> read,),
    group["t_steady_state"] |> read,
    group["one_point_r"] |> read,
    group["two_point_r"] |> read,
    group["one_point_k"] |> read,
    group["two_point_k"] |> read,
    group["n_ave"][1],
    group["window1"] |> read,
    group["window2"] |> read,
    group["first_idx1"] |> read,
    group["first_idx2"] |> read
end
##
tspan = (0.0f0, 50.0f0) .+ t_steady_state

one_point_r, two_point_r, one_point_k, two_point_k, n_ave = update_correlations!(
    one_point_r, two_point_r, one_point_k, two_point_k, n_ave, steady_state, window1, window2, first_idx1, first_idx2, (param.L,), 10^3, 1, tspan, param.dt;
    dispersion, potential, nonlinearity, pump, param, noise_func, show_progress=true,
    max_datetime=DateTime(2025, 4, 9, 10, 0));

h5open(saving_path, "cw") do file
    group = file[group_name]
    group["one_point_r"][:, :, :, :] = Array(one_point_r)
    group["two_point_r"][:, :] = Array(two_point_r)
    group["one_point_k"][:, :, :, :] = Array(one_point_k)
    group["two_point_k"][:, :] = Array(two_point_k)
    group["n_ave"][:] = [n_ave]
end