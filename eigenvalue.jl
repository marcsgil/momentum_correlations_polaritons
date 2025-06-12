using CairoMakie, JLD2, LinearAlgebra
include("polariton_funcs.jl")

saving_dir = "/home/marcsgil/Code/LEON/MomentumCorrelations/full_sim2/"

steady_state, param, t_steady_state = jldopen(joinpath(saving_dir, "steady_state.jld2")) do file
    file["steady_state"],
    file["param"],
    file["t_steady_state"]
end

xs = StepRangeLen(0, param.dx, param.N - 1) .- param.x_def

idx_min = findfirst(x -> x >= -100, xs)
idx_max = findlast(x -> x <= 100, xs)
idxs = idx_min:idx_max

gns = param.g * abs2.(steady_state[1][2:end])[idxs]
ks = diff(unwrap(angle.(steady_state[1])))[idxs] / param.dx
vs = param.ħ * ks / param.m
δs = [detuning(param.δ₀, k, param.ħ, param.m) for k in ks]

Γ = param.γ .+ vcat(0, diff(vs))



second_derivative_operator(100, 1)