using GeneralizedGrossPitaevskii, CUDA, CairoMakie
include("../polariton_funcs.jl")
include("../io.jl")
include("equations.jl")

# Space parameters
L = 1600.0f0
lengths = (L,)
N = 1024
δL = L / N
rs = range(; start=-L / 2, step=δL, length=N)
ks = range(; start=-π / δL, step=2π / L, length=N)

# Polariton parameters
ħ = 0.6582f0 #meV.ps
γ = 0.047f0 / ħ
m = 1 / 6f0 # meV.ps^2/μm^2; This is 3×10^-5 the electron mass
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
t_freeze = 288.0f0

δt = 2.0f-1

# Full parameter tuple
param = (; δ₀, m, γ, ħ, L, g, V_damp, w_damp, V_def, w_def,
    Amax, t_cycle, t_freeze, δL, N, k_down, k_up, divide, factor, δt)

u0 = (CUDA.zeros(N), CUDA.zeros(N))
prob = GrossPitaevskiiProblem(u0, lengths; dispersion, potential, nonlinearity, pump, param)
tspan = (0, 1200.0f0)
solver = StrangSplittingC(512, δt)
ts, sol = GeneralizedGrossPitaevskii.solve(prob, solver, tspan);

steady_state = map(x -> Array(x[:, end]), sol)

heatmap(rs, ts, Array(real(first(sol) .* last(sol))))
##
n = real(first(steady_state) .* last(steady_state))
n_up = n[N÷4]
n_down = n[3N÷4]

with_theme(theme_latexfonts()) do
    fig = Figure(; fontsize=20)
    ax = Axis(fig[1, 1], xlabel=L"x", ylabel=L"gn")
    offset = 300
    J = N÷2-offset:N÷2+offset
    lines!(ax, rs[J], g * n[J], linewidth=4)
    fig
end
##
v = velocity(first(steady_state), ħ, m, δL)
c = map((n, v) -> speed_of_sound(n, g, δ₀, m * v / ħ, ħ, m), n, v)

with_theme(theme_latexfonts()) do
    fig = Figure(; fontsize=20)
    ax = Axis(fig[1, 1], xlabel=L"x")
    offset = 100
    J = N÷2-offset:N÷2+offset
    lines!(ax, rs[J], c[J], linewidth=4, color=:blue, label=L"c")
    lines!(ax, rs[J], v[J], linewidth=4, color=:red, label=L"v")
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
    two_point = zero(steady_state[1]) * zero(steady_state[1])'
    one_point = stack(two_point for a ∈ 1:2)
    one_point, two_point
end

function create_save_group(_steady_state, saving_path, group_name, win_func1, param1, win_func2, param2)
    steady_state = Array.(_steady_state)

    one_point_r, two_point_r = get_correlation_buffers(steady_state)
    one_point_k, two_point_k = get_correlation_buffers(steady_state)


    kernel1 = (similar(two_point_k), similar(two_point_k))
    kernel2 = similar.(kernel1)

    GeneralizedGrossPitaevskii.grid_map!(kernel1[1], args -> cis(-prod(args)) * win_func1(args..., param1), (ks, rs))
    GeneralizedGrossPitaevskii.grid_map!(kernel2[1], args -> cis(-prod(args)) * win_func2(args..., param2), (ks, rs))
    GeneralizedGrossPitaevskii.grid_map!(kernel1[2], args -> cis(prod(args)) * conj(win_func1(args..., param1)), (ks, rs))
    GeneralizedGrossPitaevskii.grid_map!(kernel2[2], args -> cis(prod(args)) * conj(win_func2(args..., param2)), (ks, rs))

    n_ave = 0

    h5open(saving_path, "cw") do file
        group = create_group(file, group_name)
        write_parameters!(group, param)
        group["steady_state"] = stack(x for x ∈ steady_state)
        group["t_steady_state"] = tspan[end]
        group["one_point_r"] = one_point_r
        group["two_point_r"] = two_point_r
        group["one_point_k"] = one_point_k
        group["two_point_k"] = two_point_k
        group["n_ave"] = [n_ave]
        group["kernel1"] = stack(x for x ∈ kernel1)
        group["kernel2"] = stack(x for x ∈ kernel2)
    end
end

saving_path = "/home/stagios/Marcos/LEON_Marcos/Users/Marcos/MomentumCorrelations/PositiveP/correlations.h5"
group_name = "test_new"

win_func1(k, x, param) = exp(-(x - 100)^2 / 100^2)
win_func2(k, x, param) = exp(-(x + 100)^2 / 100^2)

#= h5open(saving_path, "cw") do file
    delete_object(file, group_name)
end =#

create_save_group(steady_state, saving_path, group_name, win_func1, nothing, win_func2, nothing)