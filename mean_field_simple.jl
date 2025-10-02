using GeneralizedGrossPitaevskii, CairoMakie
include("polariton_funcs.jl")
include("io.jl")
include("equations_simple.jl")
include("plot_funcs.jl")

# Space parameters
L = 512.0
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
V_damp = 50.0
w_damp = 20.0
V_pot = 1.3
w_pot = 0.75

# Pump parameters
w_pump = w_damp * 2
extra_space_amplitude = 20

k_up = 2π * 12 / L
k_down = 2π * 48 / L

δ_up = δ₀ - ħ * k_up^2 / 2m
δ_down = δ₀ - ħ * k_down^2 / 2m

F_sonic_up = γ * √(δ_up / g) / 2
F_sonic_down = γ * √(δ_down / g) / 2

F_up = F_sonic_up * (1 + 0.19)
F_down = F_sonic_down * (1 + 0.6)

decay_time = 50.0
extra_time_amplitude = 5.0

dt = 2.0e-2
nsaves = 256

# Full parameter tuple
param = (;
    L, N, dx, dt,
    m, g, ħ, γ, δ₀,
    V_damp, w_damp, w_pump, V_pot, w_pot, extra_space_amplitude,
    k_up, k_down, F_up, F_down, decay_time, extra_time_amplitude
)

u0 = (zeros(complex(typeof(L)), N),)
prob = GrossPitaevskiiProblem(u0, lengths; dispersion, nonlinearity, pump, potential, param)
tspan = (0, 600.0)
alg = StrangSplitting()
ts, sol = solve(prob, alg, tspan; dt, nsaves);
steady_state = map(x -> x[:, end], sol)
heatmap(xs .- L / 2, ts, Array(abs2.(sol[1])))
plot_velocities(xs .- param.L / 2, steady_state[1], param; xlims=(-130, 130), ylims=(0, 2.5))
##
n0 = abs2.(sol[1][7N÷8, end])
ns = LinRange(0, 1.3 * n0, 256)
Is = eq_of_state.(ns, g, δ₀, k_down, ħ, m, γ)

with_theme(theme_latexfonts()) do
    fig = Figure()
    ax = Axis(fig[1, 1])
    lines!(ax, Is, ns)
    scatter!(ax, abs2(F_down), n0, color=:black)
    fig
end
##
saving_dir = "/home/marcsgil/Data/momentum_correlation_polaritons/simple"
save_steady_state(saving_dir, steady_state, param, tspan)