using Revise, GeneralizedGrossPitaevskii, CairoMakie, LinearAlgebra, CUDA, Statistics, ProgressMeter, KernelAbstractions, FFTW
include("polariton_funcs.jl")

function dispersion(ks, param)
    -im * param.γ / 2 + param.ħ * sum(abs2, ks) / 2param.m - param.δ₀
end

function amplitude(t, Amax, t_cycle, t_freeze)
    _t = ifelse(t > t_freeze, t_freeze, t)
    val = Amax * _t * (t_cycle - _t) * 4 / t_cycle^2
    val < 0 ? zero(val) : val
end

function damping_potential(x::NTuple{1}, xmin, xmax, width)
    -im * exp(-(x[1] - xmin)^2 / width^2) + exp(-(x[1] - xmax)^2 / width^2)
end

function potential(rs, param)
    param.V_def * exp(-sum(abs2, rs) / param.w_def^2) +
    param.V_damp * damping_potential(rs, -param.L / 2, param.L / 2, param.w_damp)
end

function phase(x, width, k_up, k_down)
    k₋ = (k_down - k_up) / 2
    k₊ = (k_down + k_up) / 2
    k₋ * width * log(cosh(x / width)) + k₊ * x
end

function pump(x, param, t)
    a = amplitude(t, param.Amax, param.t_cycle, param.t_freeze)
    if x[1] ≤ -0.9 * param.waist
        a *= 4
    elseif x[1] ≥ 0
        a *= 1
    end
    ϕ = phase(x[1], param.width, param.k_up, param.k_down)
    a * (abs(x[1]) ≤ param.waist) * cis(ϕ)
end

noise_func(ψ, param) = √(param.γ / 2 / param.δL)

nonlinearity(ψ, param) = param.g * abs2(ψ[1])

# Space parameters
L = 1600.0f0
lengths = (L,)
N = 1024
δL = L / N
rs = range(; start=-L / 2, step=L / N, length=N)

# Polariton parameters
ħ = 0.6582f0 #meV.ps
γ = 0.047f0 / ħ
m = 7 / 16f0
g = 0.0003f0 / ħ
δ₀ = 0.2 / ħ

# Potential parameters
V_damp = 100.0f0
w_damp = 5.0f0
V_def = 0f0#-0.85f0 / ħ
w_def = 0.75f0

# Pump parameters
k_up = 0.1f0
k_down = 0.5f0

δ_up = δ₀ - ħ * k_up^2 / 2m
δ_down = δ₀ - ħ * k_down^2 / 2m

waist = 600f0
width = 18f0

# Bistability cycle parameters
Imax = 90.0f0
Amax = √Imax
t_cycle = 1000.0f0
t_freeze = 970.0f0

dt = 5f-2

# Full parameter tuple
param = (; δ₀, m, γ, ħ, L, g,
    Amax, t_cycle, t_freeze, δL, waist, width, k_up, k_down,
    V_damp, w_damp, V_def, w_def)

u0_empty = (zeros(ComplexF32, N), )
prob_steady = GrossPitaevskiiProblem(u0_empty, lengths; dispersion, nonlinearity, pump, param)
tspan_steady = (0, 2000.0f0)
alg = StrangSplittingC()
nsaves = 512
ts_steady, sol_steady = GeneralizedGrossPitaevskii.solve(prob_steady, alg, tspan_steady; dt, nsaves);

steady_state = sol_steady[1][:, end]
heatmap(rs, ts_steady, Array(abs2.(sol_steady[1])))
##
n = Array(abs2.(steady_state))
n_up = n[N÷4]
n_down = n[3N÷4]

with_theme(theme_latexfonts()) do
    fig = Figure(; fontsize=20)
    ax = Axis(fig[1, 1], xlabel=L"x", ylabel=L"gn", xticks=(-800:200:800))
    #xlims!(ax, -200, 200)
    #ylims!(ax, -0.01, 0.75)
    lines!(ax, rs, g * n, linewidth=4)
    #save("/home/stagios/Marcos/LEON_Marcos/Users/Marcos/MomentumCorrelations/Plots/TruncatedWigner/densities.pdf", fig)
    fig
end
##
v = velocity(Array(steady_state), ħ, m, δL)
c = map((n, v) -> speed_of_sound(n, g, δ₀, m * v / ħ, ħ, m), n, v)

with_theme(theme_latexfonts()) do
    fig = Figure(; fontsize=20)
    ax = Axis(fig[1, 1], xlabel=L"x")
    xlims!(ax, -200, 200)
    ylims!(ax, 0, 1.5)
    lines!(ax, rs, c, linewidth=4, color=:blue, label=L"c")
    lines!(ax, rs, v, linewidth=4, color=:red, label=L"v")
    axislegend(; position=:lt)
    #save("/home/stagios/Marcos/LEON_Marcos/Users/Marcos/MomentumCorrelations/Plots/TruncatedWigner/velocities.pdf", fig)
    fig
end
##
ns_up_theo = LinRange(0, 1000, 512)
Is_up_theo = eq_of_state.(ns_up_theo, g, δ₀, k_up, ħ, m, γ)

γ^2 * detuning(δ₀, k_up, ħ, m) / 4g

ns_down_theo = LinRange(0, 600, 512)
Is_down_theo = eq_of_state.(ns_down_theo, g, δ₀, k_down, ħ, m, γ)

with_theme(theme_latexfonts()) do
    fig = Figure(fontsize=16)
    ax = Axis(fig[1, 1]; xlabel="I", ylabel="n")
    lines!(ax, Is_up_theo, ns_up_theo, color=:blue, linewidth=4, label="Upstream")
    lines!(ax, Is_down_theo, ns_down_theo, color=:red, linewidth=4, label="Downstream")
    A_stop = amplitude(Inf, param.Amax, param.t_cycle, param.t_freeze)
    scatter!(ax, abs2(A_stop), n_up, color=:black, markersize=16)
    scatter!(ax, abs2(A_stop), n_down, color=:black, markersize=16)
    #vlines!(ax, abs2(A_stop))
    axislegend()
    #save("/home/stagios/Marcos/LEON_Marcos/Users/Marcos/MomentumCorrelations/Plots/TruncatedWigner/bistability.pdf", fig)
    fig
end