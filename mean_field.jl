using GeneralizedGrossPitaevskii, CairoMakie
include("polariton_funcs.jl")
include("io.jl")
include("equations.jl")
include("plot_funcs.jl")

# Space parameters
L = 1024.0
lengths = (L,)
N = 512
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
w_damp = 20.0
x_def = L / 2
V_def = 0.85 / ħ
w_def = 0.75

# Pump parameters
k_up = 0.148
k_down = 0.614

divide = x_def - 7

δ_up = δ₀ - ħ * k_up^2 / 2m
δ_down = δ₀ - ħ * k_down^2 / 2m

F_sonic_up = γ * √(δ_up / g) / 2
F_sonic_down = γ * √(δ_down / g) / 2

F_up = F_sonic_up + 0.01
F_down = F_sonic_down + 0.4
F_max = 20

w_pump = 20

decay_time = 50.0
extra_intensity = 6.0

dt = 2.0e-1
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

plot_velocities(xs .- x_def, steady_state[1], param; xlims=(-150, 150), ylims=(0, 3))
#plot_bistability(xs .- x_def, steady_state[1], param, -200, 200, factor_ns_down=1.2)
##
saving_dir = "/home/stagios/Marcos/LEON_Marcos/MomentumCorrelations/test"

plot_density(xs, steady_state[1], param; saving_dir)
plot_velocities(xs .- x_def, steady_state[1], param; xlims=(-100, 100), ylims=(0, 3), saving_dir)
plot_bistability(xs .- x_def, steady_state[1], param, -100, 130; saving_dir, factor_ns_down=1.3)

ks_up = LinRange(-0.7, 0.7, 512)
ks_down = LinRange(-1.5, 1.5, 512)
plot_dispersion(xs .- x_def, steady_state[1], param, -200, 200, 0.4, ks_up, ks_down; saving_dir)
##
save_steady_state(saving_dir, steady_state, param, tspan)