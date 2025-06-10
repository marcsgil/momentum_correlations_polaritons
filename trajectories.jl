using CairoMakie, JLD2, Integrals, Interpolations, Roots, Polynomials
include("polariton_funcs.jl")

saving_dir = "/home/marcsgil/Code/LEON/MomentumCorrelations/full_sim2/"

steady_state, param, t_steady_state = jldopen(joinpath(saving_dir, "steady_state.jld2")) do file
    file["steady_state"],
    file["param"],
    file["t_steady_state"]
end

J = (-100:100) .+ 256

xs = (StepRangeLen(0, param.dx, param.N - 1) .- param.x_def)

ns = abs2.(steady_state[1][2:end])
v = velocity(Array(steady_state[1]), param.ħ, param.m, param.dx)
ks = diff(unwrap(angle.(steady_state[1]))) / param.dx
δs = param.δ₀ .- param.ħ * ks.^2 / (2 * param.m)
c = map((ψ, k) -> speed_of_sound(abs2(ψ), param.g, param.δ₀, k, param.ħ, param.m), Array(steady_state[1])[begin+1:end], ks)

lines(xs[J], v[J])



v_itp = linear_interpolation(xs[J], v[J], extrapolation_bc=Flat())
n_itp = linear_interpolation(xs[J], ns[J], extrapolation_bc=Flat())
δ_itp = linear_interpolation(xs[J], δs[J], extrapolation_bc=Flat())

K = findall(!isnan, c[J])
c_itp = linear_interpolation(xs[J][K], c[J][K], extrapolation_bc=Flat())

new_xs = LinRange(-100, 100, 2000)
lines(new_xs, δ_itp(new_xs), color=:red, linewidth=2)
##

param

function find_qs(x, Ω, v, c, n, δ, param)
    mass_term = (3param.g * n(x) - δ(x)) * (param.g * n(x) - δ(x))
    roots(Polynomial([mass_term - Ω^2, -2v(x) * Ω, c(x)^2 - v(x)^2, 0, (param.ħ / 2 / param.m)^2]))
end

find_qs(Inf, 0.64, v_itp, c_itp, n_itp, δ_itp, param)