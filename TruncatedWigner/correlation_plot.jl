using CairoMakie, HDF5, FFTW
include("../tracing.jl")
include("../io.jl")
include("../polariton_funcs.jl")
include("equations.jl")

saving_path = "/home/stagios/Marcos/LEON_Marcos/Users/Marcos/MomentumCorrelations/TruncatedWigner/correlations.h5"
group_name = "support_downstream"

param, steady_state, t_steady_state, one_point_r, two_point_r, one_point_k, two_point_k, kernel1, kernel2 = h5open(saving_path) do file
    group = file[group_name]

    n_ave = group["n_ave"][1]
    order_of_magnitude = round(Int, log10(n_ave))
    @info "Average after $(n_ave / 10^order_of_magnitude) × 10^$order_of_magnitude realizations"

    read_parameters(group),
    group["steady_state"] |> read,
    group["t_steady_state"] |> read,
    group["one_point_r"] |> read,
    group["two_point_r"] |> read,
    group["one_point_k"] |> read,
    group["two_point_k"] |> read,
    group["kernel1"] |> read,
    group["kernel2"] |> read,
    group["ks"] |> read
end

commutators_r = calculate_position_commutators(one_point_r, param.δL)
commutators_k = calculate_momentum_commutators(kernel1, kernel2, param.L)
g2_r = calculate_g2(one_point_r, two_point_r, commutators_r)
g2_k = calculate_g2(one_point_k, two_point_k, commutators_k)


N = param.N
L = param.L
δL = param.δL
rs = range(; start=-param.L / 2, step=param.δL, length=param.N)

m = param.m
δ₀ = param.δ₀
g = param.g
ħ = param.ħ
k_up = param.k_up
k_down = param.k_down
x_def = param.x_def

n = Array(abs2.(steady_state))

n_up = n[argmin(abs.(rs .- x_def .+ 100))]
n_down = n[argmin(abs.(rs .- x_def .- 200))]

power = 4
with_theme(theme_latexfonts()) do
    fig = Figure(; size=(730, 600), fontsize=20)
    ax = Axis(fig[1, 1], aspect=DataAspect(), xlabel=L"x", ylabel=L"x\prime")
    xlims!(ax, (-100, 100) .+ param.x_def)
    ylims!(ax, (-100, 100) .+ param.x_def)
    hm = heatmap!(ax, rs, rs, (g2_r .- 1) * 10^power, colorrange=(-1, 1), colormap=:inferno)
    Colorbar(fig[1, 2], hm, label=L"g_2(x, x\prime) -1 \ \ ( \times 10^{-%$power})")
    #save("/home/stagios/Marcos/LEON_Marcos/Users/Marcos/MomentumCorrelations/Plots/TruncatedWigner/g2_postion.pdf", fig)
    fig
end
##
with_theme(theme_latexfonts()) do
    fig = Figure(; fontsize=20)
    ax = Axis(fig[1, 1], xlabel=L"x \ (\mu m)", xticks=-800:200:800)
    lines!(ax, rs, g .* abs2.(steady_state), linewidth=4, label=L"gn")
    lines!(ax, rs, abs.(kernel1)[1, :], linewidth=4, linestyle=:dash, label="Window 1 (a.u.)")
    lines!(ax, rs, abs.(kernel2)[1, :], linewidth=4, linestyle=:dash, label="Window 2 (a.u.)")
    axislegend(ax)
    #save("/home/stagios/Marcos/LEON_Marcos/Users/Marcos/MomentumCorrelations/Plots/TruncatedWigner/windows_100.pdf", fig)
    fig
end
##
param_up = (n_up, param.g, param.δ₀, param.k_up, param.ħ, param.m)
param_down = (n_down, param.g, param.δ₀, param.k_down, param.ħ, param.m)

#Hawking (u1 out, d2 out)
param1 = (param_up..., true)
param2 = (param_down..., false)

k_ext1, ω_ext1 = find_extrema(dispersion_relation, (-1, 1), param1...)
k_ext2, ω_ext2 = find_extrema(dispersion_relation, (0.1, 1), param2...)

k1_min = find_zero(k -> dispersion_relation(k, param1...) - ω_ext2, (-1, k_ext1))
k2_min = find_zero(k -> dispersion_relation(k, param2...) - ω_ext1, (0, k_ext2))

bracket1 = (k1_min * 0.9999, k_ext1)
bracket2 = (k2_min, k_ext2)

corr_up_u1d2, corr_down_u1d2 = correlate(param1, bracket1, param2, bracket2, 128, false)
## (u out, d1 out)
param1 = (param_up..., true)
param2 = (param_down..., true)

k_ext1, ω_ext1 = find_extrema(dispersion_relation, (-1, 1), param1...)

k1_min = find_zero(k -> dispersion_relation(k, param1...) - ω_ext2, (-1, k_ext1))

bracket1 = (-0.8, k_ext1)
bracket2 = (0.01, 1)

corr_up_u1d1, corr_down_u1d1 = correlate(param1, bracket1, param2, bracket2, 128, false)
## (d1, d2)
param1 = (param_down..., true)
param2 = (param_down..., false)

k_ext1, ω_ext1 = find_extrema(dispersion_relation, (-1, -0.1), param1...)
k_ext2, ω_ext2 = find_extrema(dispersion_relation, (0.1, 1), param2...)

k1_max = find_zero(k -> dispersion_relation(k, param1...) - ω_ext2, (0.1, 1))
k2_min = find_zero(k -> dispersion_relation(k, param2...) - ω_ext1, (-1, -0.1))

bracket1 = (0, k1_max)
bracket2 = (0, k_ext2)

corr_d1d2, corr_d1d2′ = correlate(param1, bracket1, param2, bracket2, 128, false)
## (d1_star, d2_star)
param1 = (param_down..., true)
param2 = (param_down..., false)

k_ext1, ω_ext1 = find_extrema(dispersion_relation, (-1, -0.1), param1...)
k_ext2, ω_ext2 = find_extrema(dispersion_relation, (0.1, 1), param2...)

k1_max = find_zero(k -> dispersion_relation(k, param1...) - ω_ext2, (0.1, 1))
k2_min = find_zero(k -> dispersion_relation(k, param2...) - ω_ext1, (-1, -0.1))

bracket1 = (k_ext1, 0)
bracket2 = (k2_min, 0)

corr_d1_star_d2_star, corr_d1_star_d2_star′ = correlate(param1, bracket1, param2, bracket2, 128, false)
## (d2, d2_star)
param1 = (param_down..., true)
param2 = (param_down..., false)

k_ext1, ω_ext1 = find_extrema(dispersion_relation, (-1, -0.1), param1...)
k_ext2, ω_ext2 = find_extrema(dispersion_relation, (0.1, 1), param2...)

k1_max = find_zero(k -> dispersion_relation(k, param1...) - ω_ext2, (0.1, 1))
k2_min = find_zero(k -> dispersion_relation(k, param2...) - ω_ext1, (-1, -0.1))

dispersion_relation(k1_max, param1...)

bracket1 = (0, k1_max)
bracket2 = (k2_min, 0)

corr_d2d2_star, corr_d2d2_star′ = correlate(param1, bracket1, param2, bracket2, 128, true)
##
ks = range(; start=-π / param.δL, step=2π / (size(g2_k, 1) * param.δL), length=size(g2_k, 1))
power = 4

xticks = [0.0, k_down]
yticks = [0.0, k_up]

xticklabels = [L"0", L"k_{d}"]
yticklabels = [L"0", L"k_{u}"]

with_theme(theme_latexfonts()) do
    fig = Figure(; size=(900, 600), fontsize=20)
    ax = Axis(fig[1, 1]; aspect=DataAspect(), xlabel=L"k", ylabel=L"k\prime", xticks=(xticks, xticklabels), yticks=(yticks, yticklabels))
    xlims!(ax, (-0.7, 1.3))
    ylims!(ax, (-0.7, 1.3))
    hm = heatmap!(ax, ks, ks, (g2_k .- 1) * 10^power, colorrange=(-3, 3), colormap=:inferno)
    Colorbar(fig[1, 2], hm, label=L"g_2(k, k\prime) -1 \ \ ( \times 10^{-%$power})")
    #lines!(ax, corr_down_u1d2 .+ k_down, corr_up_u1d2 .+ k_up, linewidth=4, color=(:green, 0.6), linestyle=:dash, label=L"u_{\text{out}} \leftrightarrow d2_{\text{out}}")
    #lines!(ax, corr_down_u1d1 .+ k_down, corr_up_u1d1 .+ k_up, linewidth=4, color=(:purple, 0.6), linestyle=:dash, label=L"u_{\text{out}} \leftrightarrow d1_{\text{out}}")
    #lines!(ax, -corr_down_u1d2 .+ k_down, corr_up_u1d2 .+ k_up, linewidth=4, color=(:red, 0.6), linestyle=:dash, label=L"u_{\text{out}} \leftrightarrow d2_{\text{out}}^*")
    #lines!(ax, -corr_down_u1d1 .+ k_down, corr_up_u1d1 .+ k_up, linewidth=4, color=(:orange, 0.6), linestyle=:dash, label=L"u_{\text{out}} \leftrightarrow d1_{\text{out}}^*")


    #lines!(ax, corr_d1d2 .+ k_down, corr_d1d2′ .+ k_down, linewidth=4, color=:green, linestyle=:dash, label=L"d1_{\text{out}} \leftrightarrow d2_{\text{out}}")
    #lines!(ax, corr_d1_star_d2_star′ .+ k_down, corr_d1_star_d2_star .+ k_down, linewidth=4, color=:purple, linestyle=:dash, label=L"d1_{\text{out}}^* \leftrightarrow d2_{\text{out}}^*")
    #lines!(ax, corr_d2d2_star .+ k_down, corr_d2d2_star′ .+ k_down, linewidth=4, color=:orange, linestyle=:dash, label=L"d1_{\text{out}} \leftrightarrow d1_{\text{out}}^*")
    #scatter!(ax, k_up - 0.3, k_up + 0.15, color=:cyan, markersize=16, label = "?")
    #Legend(fig[1, 3], ax)

    #save("/home/stagios/Marcos/LEON_Marcos/Users/Marcos/MomentumCorrelations/Plots/TruncatedWigner/g2_momentum_150.pdf", fig)
    fig
end
##
#diag_part = diag((g2_k .- 1) * 10^power)
freq_freq = rfftfreq(length(ks), 2π / (ks[2] - ks[1]))


g2_k
J = 370:640

heatmap((g2_k[J, J] .- 1) * 10^power, colorrange=(-2, 2), colormap=:inferno)

heatmap(freq_freq[1:25], freq_freq[1:25], abs.(rfft(g2_k[J, J] .- 1) * 10^power)[1:25, 1:25], colormap=:inferno)

with_theme(theme_latexfonts()) do
    fig = Figure()
    ax = Axis(fig[1, 1])
    lines!(diag_part)
    fig
end

with_theme(theme_latexfonts()) do
    fig = Figure()
    ax = Axis(fig[1, 1])
    xlims!(ax, (0, 50))
    lines!(ax, freq_freq, abs.(rfft(diag_part)))
    fig
end