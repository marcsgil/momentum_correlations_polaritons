using GeneralizedGrossPitaevskii, CairoMakie, CUDA
include("../polariton_funcs.jl")
include("../io.jl")
include("equations_new.jl")

# Space parameters
L = 2048.0f0
lengths = (L,)
N = 1024
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
V_damp = 4.5f0 / ħ
w_damp = 20.0f0
x_def = 0.0f0
V_def = -0.85f0 / ħ
w_def = 0.75f0

# Pump parameters
k_up = 0.27f0
k_down = 0.585f0

divide = x_def - 7f0

δ_up = δ₀ - ħ * k_up^2 / 2m
δ_down = δ₀ - ħ * k_down^2 / 2m

F_sonic_up = γ * √(δ_up / g) / 2
F_sonic_down = γ * √(δ_down / g) / 2

F_up = F_sonic_up + 0.0f0
F_down = F_sonic_down + 0.030
F_max = 11f0

w_pump = 30f0

dt = 2.0f-1
nsaves = 512

# Full parameter tuple
param = (;
    L, N, δL, dt,
    m, g, ħ, γ, δ₀,
    V_damp, w_damp, V_def, w_def, x_def,
    k_up, k_down, divide, F_up, F_down, F_max, w_pump
)


u0 = (CUDA.zeros(complex(typeof(L)), N),)
prob = GrossPitaevskiiProblem(u0, lengths; dispersion, potential, nonlinearity, pump, param)
tspan = (0f0, 1000.0f0)
alg = StrangSplittingC()
ts, sol = GeneralizedGrossPitaevskii.solve(prob, alg, tspan; dt, nsaves);
steady_state = sol[1][:, end]
heatmap(rs, ts, Array(abs2.(sol[1])))
##
function plot_density!(ax, rs, field, param; xlims=nothing, ylims=nothing)
    n = Array(abs2.(field))
    lines!(ax, rs, param.g * n, linewidth=4)
    !isnothing(xlims) && xlims!(ax, xlims...)
    !isnothing(ylims) && ylims!(ax, ylims...)
end

function plot_density(rs, field, param; xlims=nothing, ylims=nothing, fontsize=20, savepath=nothing)
    with_theme(theme_latexfonts()) do
        fig = Figure(; fontsize)
        ax = Axis(fig[1, 1], xlabel=L"x", ylabel=L"gn")
        plot_density!(ax, rs, field, param; xlims, ylims)
        !isnothing(savepath) && save(savepath, fig)
        fig
    end
end

plot_density(rs, steady_state, param;)
##

function plot_velocities!(ax, rs, field, param; xlims=nothing, ylims=nothing)
    v = velocity(Array(field), param.ħ, param.m, param.δL)
    c = Array(@. sqrt(ħ * g * abs2(field) / m))

    !isnothing(xlims) && xlims!(ax, xlims...)
    !isnothing(ylims) && ylims!(ax, ylims...)
    lines!(ax, rs, c, linewidth=4, color=:blue, label=L"c")
    lines!(ax, rs[begin+1:end], v, linewidth=4, color=:red, label=L"v")
    axislegend(; position=:lt)
end

function plot_velocities(rs, field, param; xlims=nothing, ylims=nothing, fontsize=20, savepath=nothing)
    with_theme(theme_latexfonts()) do
        fig = Figure(; fontsize)
        ax = Axis(fig[1, 1], xlabel=L"x")
        plot_velocities!(ax, rs, field, param; xlims, ylims)
        fig
    end
end

plot_velocities(rs, steady_state, param; xlims=(-50, 50), ylims=(0, 5))
##
ns_up_theo = LinRange(0, 1500, 512)
Is_up_theo = eq_of_state.(ns_up_theo, g, δ₀, k_up, ħ, m, γ)

ns_down_theo = LinRange(0, 500, 512)
Is_down_theo = eq_of_state.(ns_down_theo, g, δ₀, k_down, ħ, m, γ)

with_theme(theme_latexfonts()) do
    fig = Figure(fontsize=16)
    ax = Axis(fig[1, 1]; xlabel="I", ylabel="n")
    xlims!(ax, 0, 10)
    lines!(ax, Is_up_theo, ns_up_theo, color=:blue, linewidth=4, label="Upstream")
    lines!(ax, Is_down_theo, ns_down_theo, color=:red, linewidth=4, label="Downstream")
    scatter!(ax, abs2(F_up), n_up, color=:black, markersize=16)
    scatter!(ax, abs2(F_down), n_down, color=:black, markersize=16)
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

function create_save_group(_steady_state, saving_path, group_name, win_func1, param1, win_func2, param2, rs, param)
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
group_name = "support_downstream"

win_func1(k, x, param) = exp(-(x - 150)^2 / 100^2)
win_func2(k, x, param) = exp(-(x + 150)^2 / 100^2)

function hamming(k, x, param)
    a0 = oftype(x, 0.54) * (abs(x - param.x₀) ≤ param.width / 2)
    a1 = oftype(x, 0.46) * (abs(x - param.x₀) ≤ param.width / 2)
    a0 + a1 * cospi(2 * (x - param.x₀) / param.width)
end

window_width = 500f0
param1 = (; x₀=x_horizon + window_width / 2 - 10, width=window_width)
param2 = (; x₀=x_horizon - window_width / 2 + 10, width=window_width)

#= h5open(saving_path, "cw") do file
    delete_object(file, group_name)
end =#

create_save_group(steady_state, saving_path, group_name, hamming, param1, hamming, param2, rs, param)