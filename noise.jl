using GeneralizedGrossPitaevskii, CUDA, Random
include("io.jl")
include("equations.jl")
include("correlation_kernels.jl")

saving_dir = "/Volumes/partages/EQ15B/LEON-15B/Users/Marcos/MomentumCorrelations/SupportDownstreamRepulsive"

param, steady_state, t_steady_state, one_point_r, two_point_r, one_point_k, two_point_k, n_ave, window1, window2, first_idx1, first_idx2 = jldopen(joinpath(saving_dir, "correlations.jld2")) do file
    file["param"],
    file["steady_state"],
    file["t_steady_state"],
    file["one_point_r"],
    file["two_point_r"],
    file["one_point_k"],
    file["two_point_k"],
    file["n_ave"][1],
    file["window1"],
    file["window2"],
    file["first_idx1"],
    file["first_idx2"]
end

tspan = (0.0f0, 50.0f0) .+ t_steady_state

one_point_r, two_point_r, one_point_k, two_point_k, n_ave = update_correlations!(
    one_point_r, two_point_r, one_point_k, two_point_k, n_ave, steady_state, window1, window2, first_idx1, first_idx2, (param.L,), 10^2, 10, tspan, param.dt;
    dispersion, potential, nonlinearity, pump, param, noise_func, show_progress=true,
    max_datetime=DateTime(2025, 4, 9, 10, 0));


new_content = Dict(
    "one_point_r" => one_point_r,
    "two_point_r" => two_point_r,
    "one_point_k" => one_point_k,
    "two_point_k" => two_point_k,
    "n_ave" => n_ave
)

create_new_then_rename(joinpath(saving_dir, "correlations.jld2"), new_content)