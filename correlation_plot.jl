using CairoMakie, JLD2, FFTW
include("tracing.jl")
include("io.jl")
include("polariton_funcs.jl")
include("equations.jl")
include("plot_funcs.jl")

saving_dir = "/Volumes/partages/EQ15B/LEON-15B/Users/Marcos/MomentumCorrelations/SupportDownstreamRepulsive"

steady_state, param, t_steady_state = jldopen(joinpath(saving_dir, "steady_state.jld2")) do file
    file["steady_state"],
    file["param"],
    file["t_steady_state"]
end

window_idx = 1

position_averages, momentum_averages = jldopen(joinpath(saving_dir, "averages.jld2")) do file
    n_ave = file["n_ave"][1]
    order_of_magnitude = round(Int, log10(n_ave))
    @info "Average after $(n_ave / 10^order_of_magnitude) × 10^$order_of_magnitude realizations"

    file["position_averages"],
    file["momentum_averages_$window_idx"]
end

lines(position_averages[2])

window1, window2, first_idx1, first_idx2 = jldopen(joinpath(saving_dir, "windows.jld2")) do file
    pair = file["window_pair_$window_idx"]
    pair.first.window,
    pair.second.window,
    pair.first.first_idx,
    pair.second.first_idx
end

commutators_r = calculate_position_commutators(param.N, param.dx)
commutators_k = calculate_momentum_commutators(window1, window2, first_idx1, first_idx2, param.dx)

#g2_r = calculate_g2m1(position_averages, commutators_r)
g2_r = position_averages[4] ./ (position_averages[1] .* position_averages[2]')

#g2_k = fftshift(calculate_g2m1(momentum_averages, commutators_k))
g2_k = fftshift( momentum_averages[4] ./ (momentum_averages[1] .* momentum_averages[2]') )

N = param.N
L = param.L
dx = param.dx
xs = StepRangeLen(0, dx, N) .- param.x_def

m = param.m
δ₀ = param.δ₀
g = param.g
ħ = param.ħ
k_up = param.k_up
k_down = param.k_down
x_def = param.x_def

n = Array(abs2.(steady_state[1]))

n_up = n[argmin(abs.(xs .+ 500))]
n_down = n[argmin(abs.(xs .- 500))]

pow = 5
with_theme(theme_latexfonts()) do
    fig = Figure(; size=(730, 600), fontsize=20)
    ax = Axis(fig[1, 1], aspect=DataAspect(), xlabel=L"x", ylabel=L"x\prime")
    xlims!(ax, (-150, 150))
    ylims!(ax, (-150, 150))
    hm = heatmap!(ax, xs, xs, (g2_r ) * 10^pow, colorrange=(-6, 6), colormap=:inferno)
    Colorbar(fig[1, 2], hm, label=L"g_2(x, x\prime) -1 \ \ ( \times 10^{-%$pow})")
    #save(joinpath(saving_dir, "g2_position.pdf"), fig)
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

bracket1 = (-1, k_ext1)
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
ks1 = fftshift(fftfreq(length(window1), 2π / dx))
ks2 = fftshift(fftfreq(length(window2), 2π / dx))

xticks = [0.0, k_down]
yticks = [0.0, k_up]

_xticklabels = [L"0", L"k_{d}"]
_yticklabels = [L"0", L"k_{u}"]

with_theme(theme_latexfonts()) do
    pow = 2
    fig = Figure(; size=(700, 600), fontsize=20)
    ax = Axis(fig[1, 1]; aspect=DataAspect(), xlabel=L"k", ylabel=L"k\prime", xticks=(xticks, _xticklabels), yticks=(yticks, _yticklabels))
    #xlims!(ax, (-0.65, 0.65) .+ k_down)
    #ylims!(ax, (-0.65, 0.65) .+ k_up)
    hm = heatmap!(ax, ks1, ks2, (g2_k) * 10^pow, colorrange=(-6, 6), colormap=:inferno)
    Colorbar(fig[1, 2], hm, label=L"g_2(k, k\prime) -1 \ \ ( \times 10^{-%$pow})")

    #= lines!(ax, corr_down_u1d1 .+ k_down, corr_up_u1d1 .+ k_up, linewidth=4, color=(:black, 0.8), linestyle=(:dash, :loose), label=L"u_{\text{out}} \leftrightarrow d1_{\text{out}}")

    lines!(ax, corr_down_u1d1 .+ k_down, -corr_up_u1d1 .+ k_up, linewidth=4, color=(:orange, 0.8), linestyle=(:dash, :loose), label=L"u_{\text{out}}^* \leftrightarrow d1_{\text{out}}")

    lines!(ax, -corr_down_u1d2 .+ k_down, corr_up_u1d2 .+ k_up, linewidth=4, color=(:green, 0.8), linestyle=(:dash, :loose), label=L"u_{\text{out}} \leftrightarrow d2_{\text{out}}^*")
    lines!(ax, -corr_down_u1d1 .+ k_down, corr_up_u1d1 .+ k_up, linewidth=4, color=(:brown, 0.8), linestyle=(:dash, :loose), label=L"u_{\text{out}} \leftrightarrow d1_{\text{out}}^*")

    lines!(ax, -corr_down_u1d1 .+ k_down, -corr_up_u1d1 .+ k_up, linewidth=4, color=(:cyan, 0.8), linestyle=(:dash, :loose), label=L"u_{\text{out}}^* \leftrightarrow d1_{\text{out}}^*")
    lines!(ax, -corr_down_u1d2 .+ k_down, -corr_up_u1d2 .+ k_up, linewidth=4, color=(:magenta, 0.8), linestyle=(:dash, :loose), label=L"u_{\text{out}}^* \leftrightarrow d1_{\text{out}}^*") =#

    #lines!(ax, corr_d1d2 .+ k_down, corr_d1d2′ .+ k_down, linewidth=4, color=:green, linestyle=:dash, label=L"d1_{\text{out}} \leftrightarrow d2_{\text{out}}")
    #lines!(ax, corr_d1_star_d2_star′ .+ k_down, corr_d1_star_d2_star .+ k_down, linewidth=4, color=:purple, linestyle=:dash, label=L"d1_{\text{out}}^* \leftrightarrow d2_{\text{out}}^*")
    #lines!(ax, corr_d2d2_star .+ k_down, corr_d2d2_star′ .+ k_down, linewidth=4, color=:orange, linestyle=:dash, label=L"d1_{\text{out}} \leftrightarrow d1_{\text{out}}^*")
    #scatter!(ax, k_up - 0.3, k_up + 0.15, color=:cyan, markersize=16, label = "?")
    #Legend(fig[1, 3], ax)

    #save(joinpath(saving_dir, "g2_momentum_$window_idx.pdf"), fig)
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