using GeneralizedGrossPitaevskii, CairoMakie
include("polariton_funcs.jl")
include("io.jl")
include("equations.jl")
include("plot_funcs.jl")
include("correlation_kernels.jl")

# Space parameters
L = 2048.0f0
lengths = (L,)
N = 1024
dx = L / N
xs = StepRangeLen(0, dx, N)

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
k_up = 0.148f0
k_down = 0.614f0

divide = x_def - 7

δ_up = δ₀ - ħ * k_up^2 / 2m
δ_down = δ₀ - ħ * k_down^2 / 2m

F_sonic_up = γ * √(δ_up / g) / 2
F_sonic_down = γ * √(δ_down / g) / 2

F_up = F_sonic_up + 0.01f0
F_down = F_sonic_down + 0.3f0
F_max = 20f0

w_pump = 20f0

decay_time = 50.0f0
extra_intensity = 6.0f0

dt = 2.0f-1
nsaves = 512

# Full parameter tuple
param = (;
    L, N, dx, dt,
    m, g, ħ, γ, δ₀,
    V_damp, w_damp, V_def, w_def, x_def,
    k_up, k_down, divide, F_up, F_down, F_max, w_pump, extra_intensity, decay_time
)

batchsize = 1000

u0 = (randn(complex(typeof(L)), N, batchsize) .* 1 / sqrt(2dx),)
noise_prototype = similar.(u0)
prob = GrossPitaevskiiProblem(u0, lengths; dispersion, potential, nonlinearity, pump, param, noise_func, noise_prototype)
tspan = (0, 400.0f0)
alg = StrangSplitting()
ts, sol = GeneralizedGrossPitaevskii.solve(prob, alg, tspan; dt, nsaves)

mean_field = map(x -> dropdims(mean(x, dims=2), dims=2), sol)
steady_state = map(x -> x[:, end], mean_field)
heatmap(xs .- x_def, ts, Array(abs2.(mean_field[1][:, :])))
plot_velocities(xs .- x_def, steady_state[1], param; xlims=(-300, 300), ylims=(0, 3))
##
plot_density(xs, steady_state[1], param)
##
function hamming(N, ::Type{T}) where {T}
    α = 25 / 46
    β = 1 - α
    [T(α - β * cospi(2 * n / N)) for n ∈ 0:N-1]
end

function hann(N, ::Type{T}) where {T}
    [T(sinpi(n / N)^2) for n ∈ 0:N-1]
end

rect(N, ::Type{T}) where {T} = ones(T, N)

function blackman_harris(N, ::Type{T}) where {T}
    a0 = 0.35875
    a1 = 0.48829
    a2 = 0.14128
    a3 = 0.01168
    [T(a0 - a1 * cospi(2n / N) + a2 * cospi(4n / N) - a3 * cospi(6n / N)) for n ∈ 0:N-1]
end

function windowed_ft(src, window)
    dest = similar(src, length(window.window), size(src, 2))
    plan = plan_fft!(dest, 1)
    windowed_ft!(dest, src, window, plan)
end

window = Window(-50, 550, xs .- x_def, blackman_harris)

ft_sol = fftshift(windowed_ft(sol[1][:, :, end], window))

N2 = dx^2 * length(window.window) / 2π / sum(abs2, window.window)

n = (dropdims(mean(abs2, ft_sol, dims=2), dims=2) .- sum(abs2, window.window) / 2param.dx) * N2
for j ∈ eachindex(n)
    n[j] *= n[j] > 0
end

ks = fftshift(fftfreq(length(window.window), 2π / param.dx))

with_theme(theme_latexfonts()) do
    fig = Figure()
    ax = Axis(fig[1, 1], yscale=log10)
    ylims!(ax, 1e0, 1e8)
    lines!(ax, ks, n)
    fig
end