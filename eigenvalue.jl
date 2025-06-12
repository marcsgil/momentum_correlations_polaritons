using CairoMakie, JLD2, LinearAlgebra
include("polariton_funcs.jl")
include("equations.jl")

function first_derivative_operator(N, dx)
    dl = fill(-1 / 2dx, N - 1)
    du = fill(1 / 2dx, N - 1)
    d = zeros(N)
    Tridiagonal(dl, d, du)
end

function second_derivative_operator(N, dx)
    d = fill(-2 / dx^2, N)
    du = fill(1 / dx^2, N - 1)
    SymTridiagonal(d, du)
end


#= saving_dir = "/Users/marcsgil/Code/momentum_correlations_polaritons/data/"

steady_state, param, t_steady_state = jldopen(joinpath(saving_dir, "steady_state.jld2")) do file
    file["steady_state"],
    file["param"],
    file["t_steady_state"]
end =#

using GeneralizedGrossPitaevskii, CairoMakie
include("polariton_funcs.jl")
include("io.jl")
include("equations.jl")
include("plot_funcs.jl")

# Space parameters
L = 256.0
lengths = (L,)
N = 1024
dx = L / N
xs = StepRangeLen(0, dx, N)

# Polariton parameters
ħ = 0.6582 #meV.ps
γ = 0.047 / ħ
m = 1 / 6 # meV.ps^2/μm^2; This is 3×10^-5 the electron mass
g = 3e-4 / ħ
δ₀ = 0.49 / ħ

# Potential parameters
V_damp = 4.5 / ħ
w_damp = 10.0
x_def = L / 2
V_def = 0.85 / ħ
w_def = 0.75

# Pump parameters
k_up = (2π / 200) * 5
k_down = (2π / 200) * 19

divide = x_def - 7

δ_up = δ₀ - ħ * k_up^2 / 2m
δ_down = δ₀ - ħ * k_down^2 / 2m

F_sonic_up = γ * √(δ_up / g) / 2
F_sonic_down = γ * √(δ_down / g) / 2

F_up = F_sonic_up + 0.1
F_down = F_sonic_down
F_max = 20

w_pump = 20

decay_time = 50.0
extra_intensity = 6.0

dt = 1.0e-2
nsaves = 512

# Full parameter tuple
param = (;
    L, N, dx, dt,
    m, g, ħ, γ, δ₀,
    V_damp, w_damp, V_def, w_def, x_def,
    k_up, k_down, divide, F_up, F_down, F_max, w_pump, extra_intensity, decay_time
)

u0 = (zeros(complex(typeof(L)), N),)
prob = GrossPitaevskiiProblem(u0, lengths; dispersion, potential, nonlinearity, pump, param)
tspan = (0, 400.0)
alg = StrangSplitting()
ts, sol = solve(prob, alg, tspan; dt, nsaves);
steady_state = map(x -> x[:, end], sol)
heatmap(xs .- x_def, ts, Array(abs2.(sol[1])))
##
idx_min = findfirst(x -> x - param.x_def >= -50, xs)
idx_max = findlast(x -> x - param.x_def <= 50, xs)
idxs = idx_min:idx_max

gns = param.g * abs2.(steady_state[1][2:end])[idxs]
ks = diff(unwrap(angle.(steady_state[1])))[idxs] / param.dx
vs = param.ħ * ks / param.m
δs = [detuning(param.δ₀, k, param.ħ, param.m) for k in ks]

Vs = [potential(x, param) for x ∈ xs[idxs] .+ param.x_def]

Γ = vcat(0, diff(vs))

N = length(gns)
∇ = first_derivative_operator(N, param.dx)
∇² = second_derivative_operator(N, param.dx)

M11 = diagm(2gns - δs + Vs + im * Γ / 2) - im * diagm(vs) * ∇ - param.ħ * ∇² / 2param.m
M22 = diagm(2gns - δs + Vs - im * Γ / 2) + im * diagm(vs) * ∇ - param.ħ * ∇² / 2param.m
M12 = diagm(gns)

M = stack([M11 M12; M12 M22])

vals, vecs = eigen(M)

vals[1:6]

scatter(real(vals))

lines(xs[idxs] .- param.x_def, abs.(vecs[1:N, 45]))

lines(xs[idxs][2:end] .- param.x_def, diff(unwrap(angle.(vecs[1:N, 40]))))
##

imin = findlast(xs[idxs][2:end] .- param.x_def .< -20)

(xs[idxs][2:end] .- param.x_def)[imin]

q = diff(unwrap(angle.(vecs[1:N, 40])))[imin]

vals[45]

vals

dispersion_relation(q, gns[imin], 1, δ₀, ks[imin], ħ, m, false)