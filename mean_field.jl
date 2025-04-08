using GeneralizedGrossPitaevskii, CairoMakie
include("polariton_funcs.jl")
include("io.jl")
include("equations.jl")
include("plot_funcs.jl")

# Space parameters
L = 2048.0f0
lengths = (L,)
N = 1024
δL = L / N
rs = StepRangeLen(0, δL, N)

# Polariton parameters
ħ = 0.6582f0 #meV.ps
γ = 0.047f0 / ħ
m = ħ^2 / (2 * 1.29f0) #1 / 6f0 # meV.ps^2/μm^2; This is 3×10^-5 the electron mass
g = 3f-4 / ħ
δ₀ = 0.49f0 / ħ

# Potential parameters
V_damp = 4.5f0 / ħ
w_damp = 20.0f0
x_def = L / 2
V_def = 0.85f0 / ħ
w_def = 0.75f0

# Pump parameters
k_up = 0.15f0
k_down = 0.61f0

divide = x_def - 7f0

δ_up = δ₀ - ħ * k_up^2 / 2m
δ_down = δ₀ - ħ * k_down^2 / 2m

F_sonic_up = γ * √(δ_up / g) / 2
F_sonic_down = γ * √(δ_down / g) / 2

F_up = F_sonic_up + 0.01f0
F_down = F_sonic_down + 0.15f0
F_max = 20f0

w_pump = 20f0

decay_time = 100.0f0
extra_intensity = 6.0f0

dt = 2.0f-1
nsaves = 512

# Full parameter tuple
param = (;
    L, N, δL, dt,
    m, g, ħ, γ, δ₀,
    V_damp, w_damp, V_def, w_def, x_def,
    k_up, k_down, divide, F_up, F_down, F_max, w_pump, extra_intensity, decay_time
)

u0 = (zeros(complex(typeof(L)), N),)
prob = GrossPitaevskiiProblem(u0, lengths; dispersion, potential, nonlinearity, pump, param)
tspan = (0f0, 1000.0f0)
alg = StrangSplitting()
ts, sol = GeneralizedGrossPitaevskii.solve(prob, alg, tspan; dt, nsaves);
steady_state = sol[1][:, end]
heatmap(rs .- x_def, ts, Array(abs2.(sol[1])))
plot_velocities(rs .- x_def, steady_state, param; xlims=(-900, 900), ylims=(0, 3))
##
plot_density(rs, steady_state, param)
plot_velocities(rs .- x_def, steady_state, param; xlims=(-100, 100), ylims=(0, 3))
plot_bistability(rs .- x_def, steady_state, param, -500, 500)

ks_up = LinRange(-1, 1, 512)
ks_down = LinRange(-1.5, 1.5, 512)
plot_dispersion(rs .- x_def, steady_state, param, -200, 200, 0.5, ks_up, ks_down)
##
using JLD2

function get_correlation_buffers(prototype1, prototype2)
    two_point = zero(prototype1) * zero(prototype2)'
    one_point = stack(two_point for a ∈ 1:2, b ∈ 1:2)
    one_point, two_point
end

function create_save_group(_steady_state, saving_dir, param, win_func, interval1, interval2)
    steady_state = Array.(_steady_state)


    δL = param.δL
    N1 = round(Int, (last(interval1) - first(interval1)) / δL)
    N2 = round(Int, (last(interval2) - first(interval2)) / δL)
    T = eltype(first(steady_state))

    first_idx1 = argmin(idx -> abs(rs[idx] - first(interval1)), eachindex(rs))
    first_idx2 = argmin(idx -> abs(rs[idx] - first(interval2)), eachindex(rs))

    window1 = win_func.(0:N1-1, N1, T)
    window2 = win_func.(0:N2-1, N2, T)

    one_point_r, two_point_r = get_correlation_buffers(first(steady_state), first(steady_state))
    one_point_k, two_point_k = get_correlation_buffers(window1, window2)

    n_ave = 0

    jldopen(joinpath(saving_dir, "correlations.jld2"), "a+") do file
        file["steady_state"] = steady_state
        file["param"] = param
        file["t_steady_state"] = tspan[end]
        file["one_point_r"] = one_point_r
        file["two_point_r"] = two_point_r
        file["one_point_k"] = one_point_k
        file["two_point_k"] = two_point_k
        file["n_ave"] = n_ave
        file["window1"] = window1
        file["window2"] = window2
        file["first_idx1"] = first_idx1
        file["first_idx2"] = first_idx2
    end
end

saving_dir = "/Volumes/partages/EQ15B/LEON-15B/Users/Marcos/MomentumCorrelations/SupportDownstreamRepulsive"

function hamming(n, N, ::Type{T}) where {T}
    T(0.54 - 0.46 * cospi(2 * n / N))
end

create_save_group((steady_state,), saving_dir, param, hamming, (-10, 790) .+ x_def, (-790, 10) .+ x_def)