using GeneralizedGrossPitaevskii, CUDA, CairoMakie
include("../polariton_funcs.jl")
include("../io.jl")
include("equations.jl")

# Space parameters
L = 1600.0f0
lengths = (L,)
N = 1024
δL = L / N
rs = range(; start=-L / 2, step=L / N, length=N)

# Polariton parameters
ħ = 0.6582f0 #meV.ps
γ = 0.047f0 / ħ
m = ħ^2 / 2.5f0
#m = 1 / 18f0
g = 0.0003f0 / ħ
δ₀ = 0.49 / ħ

# Potential parameters
V_damp = 100.0f0
w_damp = 10.0f0
V_def = 0.85f0 / ħ
w_def = 0.75f0

# Pump parameters
k_up = 0.25f0
k_down = 0.55f0
divide = -7
factor = 0

# Bistability cycle parameters
Imax = 90.0f0
Amax = √Imax
t_cycle = 300.0f0
t_freeze = 260.0f0

δt = 2.0f-1

# Full parameter tuple
param = (; δ₀, m, γ, ħ, L, g, V_damp, w_damp, V_def, w_def,
    Amax, t_cycle, t_freeze, δL, N, k_down, k_up, divide, factor, δt)

u0 = CUDA.zeros(ComplexF32, N)
prob = GrossPitaevskiiProblem(u0, lengths; dispersion, potential, nonlinearity, pump, param)
tspan = (0, 800.0f0)
solver = StrangSplittingC(512, δt)
ts, sol = GeneralizedGrossPitaevskii.solve(prob, solver, tspan);
heatmap(rs, ts, Array(abs2.(sol)))
##
steady_state = sol[:, end]
n = Array(abs2.(steady_state))
n_up = n[N÷4]
n_down = n[3N÷4]

with_theme(theme_latexfonts()) do
    fig = Figure(; fontsize=20)
    ax = Axis(fig[1, 1], xlabel=L"x", ylabel=L"gn")
    #xlims!(ax, -200, 200)
    #ylims!(ax, -0.01, 0.8)
    lines!(ax, rs, g * n, linewidth=4)
    fig
end
##
v = velocity(Array(steady_state), ħ, m, δL)
c = map((n, v) -> speed_of_sound(n, g, δ₀, m * v / ħ, ħ, m), n, v)

with_theme(theme_latexfonts()) do
    fig = Figure(; fontsize=20)
    ax = Axis(fig[1, 1], xlabel=L"x")
    xlims!(ax, -200, 200)
    ylims!(ax, 0, 2.5)
    lines!(ax, rs, c, linewidth=4, color=:blue, label=L"c")
    lines!(ax, rs, v, linewidth=4, color=:red, label=L"v")
    axislegend(; position=:lt)
    fig
end
##
ns_up_theo = LinRange(0, 1600, 512)
Is_up_theo = eq_of_state.(ns_up_theo, g, δ₀, k_up, ħ, m, γ)

ns_down_theo = LinRange(0, 800, 512)
Is_down_theo = eq_of_state.(ns_down_theo, g, δ₀, k_down, ħ, m, γ)

with_theme(theme_latexfonts()) do
    fig = Figure(fontsize=16)
    ax = Axis(fig[1, 1]; xlabel="I", ylabel="n")
    lines!(ax, Is_up_theo, ns_up_theo, color=:blue, linewidth=4, label="Upstream")
    lines!(ax, Is_down_theo, ns_down_theo, color=:red, linewidth=4, label="Downstream")
    A_stop = A(Inf, param.Amax, param.t_cycle, param.t_freeze)
    scatter!(ax, abs2(A_stop), n_up, color=:black, markersize=16)
    scatter!(ax, abs2(A_stop * factor), n_down, color=:black, markersize=16)
    axislegend()
    fig
end
##
function get_correlation_buffers(steady_state)
    one_point = real(zero(steady_state))
    two_point = one_point * one_point'
    one_point, two_point
end

one_point_r, two_point_r = get_correlation_buffers(steady_state)
one_point_k, two_point_k = get_correlation_buffers(steady_state)

n_ave = 0

saving_path = "/home/stagios/Marcos/LEON_Marcos/Users/Marcos/MomentumCorrelations/TruncatedWigner/correlations.h5"
group_name = "no_support_gaussian"

h5open(saving_path, "cw") do file
    group = create_group(file, group_name)
    write_parameters!(group, param)
    group["steady_state"] = Array(steady_state)
    group["t_steady_state"] = tspan[end]
    group["one_point_r"] = Array(one_point_r)
    group["two_point_r"] = Array(two_point_r)
    group["one_point_k"] = Array(one_point_k)
    group["two_point_k"] = Array(two_point_k)
    group["n_ave"] = [n_ave]
end