using GeneralizedGrossPitaevskii, CairoMakie, CUDA
include("../polariton_funcs.jl")
include("../io.jl")
include("equations.jl")

# Space parameters
L = 2000.0f0
lengths = (L,)
N = 2048
δL = L / N
rs = range(; start=-L / 2, step=δL, length=N)
ks = range(; start=-π / δL, step=2π / L, length=N)

# Polariton parameters
ħ = 0.6582f0 #meV.ps
γ = 0.047f0 / ħ
m = ħ^2 / (2 * 1.29f0) #1 / 6f0 # meV.ps^2/μm^2; This is 3×10^-5 the electron mass
g = 3f-4 / ħ
δ₀ = 0.49f0 / ħ

# Potential parameters
V_damp = 1000.0f0
w_damp = 30.0f0
V_def = -0.85f0 / ħ
w_def = 0.75f0

# Pump parameters
k_up = 0.27f0
k_down = 0.539f0 #sqrt(2m * δ₀ / ħ) * 0.8f0

divide = -7
w_pump = 5f0

# Bistability cycle parameters
Amax = 4f0
#γ * sqrt((δ₀ - ħ * k_up^2) / g) / 2
Atarget_up = #1.307f0 #γ * sqrt((δ₀ - ħ * k_up^2) / g) / 2
Atarget_down = 0.8297f0
factor = Atarget_down / Atarget_up
t_cycle = 600.0f0
#t_freeze = 570.0f0

dt = 5.0f-2
nsaves = 512

# Full parameter tuple
param = (; δ₀, m, γ, ħ, L, g, V_damp, w_damp, V_def, w_def,
    Amax, t_cycle, Atarget = Atarget_up, δL, N, k_down, k_up, divide, factor, dt, w_pump)


u0 = (CUDA.zeros(complex(typeof(L)), N),)
prob = GrossPitaevskiiProblem(u0, lengths; dispersion, potential, nonlinearity, pump, param)
tspan = (0f0, 1200.0f0)
alg = StrangSplittingC()
ts, sol = GeneralizedGrossPitaevskii.solve(prob, alg, tspan; dt, nsaves);
steady_state = map(x -> x[:, end], sol)
heatmap(rs, ts, Array(abs2.(sol[1])))
##
n = Array(abs2.(steady_state[1]))
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

pump_vector = map(x -> pump((x,), param, t_cycle), rs)

with_theme(theme_latexfonts()) do
    fig = Figure(; fontsize=20)
    ax = Axis(fig[1, 1], xlabel=L"x", ylabel=L"A")
    lines!(ax, rs, abs.(pump_vector), linewidth=4)
    #hlines!(ax, Atarget_up, color=:red)
    #hlines!(ax, Atarget_down, color=:blue)
    fig
end


##
v = velocity(Array(steady_state[1]), ħ, m, δL)
c = map((n, v) -> speed_of_sound(n, g, δ₀, m * v / ħ, ħ, m), n, v)

v_pred = sqrt(2ħ * δ₀ / m)

with_theme(theme_latexfonts()) do
    fig = Figure(; fontsize=20)
    ax = Axis(fig[1, 1], xlabel=L"x")
    xlims!(ax, -200, 200)
    ylims!(ax, 0, 3)
    lines!(ax, rs, c, linewidth=4, color=:blue, label=L"c")
    lines!(ax, rs, v, linewidth=4, color=:red, label=L"v")
    #hlines!(ax, v_pred)
    axislegend(; position=:lt)
    #save("/home/stagios/Marcos/LEON_Marcos/Users/Marcos/MomentumCorrelations/Plots/TruncatedWigner/velocities.pdf", fig)
    fig
end
##
ns_up_theo = LinRange(0, 1500, 512)
Is_up_theo = eq_of_state.(ns_up_theo, g, δ₀, k_up, ħ, m, γ)

ns_down_theo = LinRange(0, 1000, 512)
Is_down_theo = eq_of_state.(ns_down_theo, g, δ₀, k_down, ħ, m, γ)

with_theme(theme_latexfonts()) do
    fig = Figure(fontsize=16)
    ax = Axis(fig[1, 1]; xlabel="I", ylabel="n")
    lines!(ax, Is_up_theo, ns_up_theo, color=:blue, linewidth=4, label="Upstream")
    lines!(ax, Is_down_theo, ns_down_theo, color=:red, linewidth=4, label="Downstream")
    scatter!(ax, abs2(Atarget), n_up, color=:black, markersize=16)
    scatter!(ax, abs2(Atarget * factor), n_down, color=:black, markersize=16)
    axislegend()
    #save("/home/stagios/Marcos/LEON_Marcos/Users/Marcos/MomentumCorrelations/Plots/TruncatedWigner/bistability.pdf", fig)
    fig
end
##
function get_correlation_buffers(steady_state)
    two_point = zero(steady_state[1]) * zero(steady_state[1])'
    one_point = stack(two_point for a ∈ 1:2, b ∈ 1:2)
    one_point, two_point
end

function create_save_group(_steady_state, saving_path, group_name, win_func1, param1, win_func2, param2, rs)
    steady_state = Array.(_steady_state)

    support = rs[findall(r -> win_func1(0, r, param1) ≠ 0, rs)]
    L_support = maximum(support) - minimum(support)
    N_support = length(support)
    δL_support = L_support / N_support
    ks = range(; start=-π / δL_support, step=2π / L_support, length=N_support)

    kernel1 = map(args -> cis(-prod(args)) * win_func1(args..., param1), Iterators.product(ks, rs))
    kernel2 = map(args -> cis(-prod(args)) * win_func2(args..., param2), Iterators.product(ks, rs))

    one_point_r, two_point_r = get_correlation_buffers(steady_state)
    one_point_k, two_point_k = get_correlation_buffers((complex(support),))

    n_ave = 0

    h5open(saving_path, "cw") do file
        group = create_group(file, group_name)
        write_parameters!(group, param)
        group["steady_state"] = stack(steady_state...)
        group["t_steady_state"] = tspan[end]
        group["one_point_r"] = one_point_r
        group["two_point_r"] = two_point_r
        group["one_point_k"] = one_point_k
        group["two_point_k"] = two_point_k
        group["n_ave"] = [n_ave]
        group["kernel1"] = kernel1
        group["kernel2"] = kernel2
        group["ks"] = Vector(ks)
    end
end

saving_path = "/home/stagios/Marcos/LEON_Marcos/Users/Marcos/MomentumCorrelations/TruncatedWigner/correlations.h5"
group_name = "support_downstream_f64"

win_func1(k, x, param) = exp(-(x - 150)^2 / 100^2)
win_func2(k, x, param) = exp(-(x + 150)^2 / 100^2)

function hamming(k, x, param)
    a0 = oftype(x, 0.54) * (abs(x - param.x₀) ≤ param.width / 2)
    a1 = oftype(x, 0.46) * (abs(x - param.x₀) ≤ param.width / 2)
    a0 + a1 * cospi(2 * (x - param.x₀) / param.width)
end

param1 = (; x₀=265, width=550)
param2 = (; x₀=-265, width=550)

support = rs[findall(r -> hamming(0, r, param1) ≠ 0, rs)]
L_support = maximum(support) - minimum(support)
N_support = length(support)
δL_support = L_support / N_support
ks = range(; start=-π / δL_support, step=2π / L_support, length=N_support)

#= h5open(saving_path, "cw") do file
    delete_object(file, group_name)
end =#

create_save_group(steady_state, saving_path, group_name, hamming, param1, hamming, param2, rs)