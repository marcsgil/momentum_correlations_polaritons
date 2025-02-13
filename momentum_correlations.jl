using CairoMakie, HDF5
include("tracing.jl")
include("io.jl")
include("polariton_funcs.jl")

G2_r, G2_k, param, steady_state = h5open("correlation.h5", "r") do file
    group = file["no_support"]
    read(group, "G2_r"), read(group, "G2_k"), read_parameters(group), read(group, "steady_state")
end

N = param.N
L = param.L
δL = param.δL
rs = range(; start=-param.L / 2, step=param.δL, length=param.N)

m = param.m
δ₀ = param.δ₀
g = param.g
ħ = param.ħ
k_up = param.k_up
k_down = √(2m * δ₀ / ħ)

n_up = abs2(Array(steady_state)[N÷4])

J = N÷2-100:N÷2+100
power = 5
with_theme(theme_latexfonts()) do
    fig = Figure(; size=(730, 600), fontsize=20)
    ax = Axis(fig[1, 1], aspect=DataAspect(), xlabel=L"x", ylabel=L"x\prime")
    hm = heatmap!(ax, rs[J], rs[J], (Array(real(G2_r)[J, J]) .- 1) * 10^power, colorrange=(-5, 5), colormap=:inferno)
    Colorbar(fig[1, 2], hm, label=L"g_2(x, x\prime) -1 \ \ ( \times 10^{-%$power})")
    fig
end
##
ks1 = LinRange(-1, 1, 512)
ks2 = LinRange(-1.5, 1.5, 512)

param_up = (n_up, g, δ₀, k_up, ħ, m)
ω₊_up = [dispersion_relation(k, param_up..., true) for k ∈ ks1]
ω₋_up = [dispersion_relation(k, param_up..., false) for k ∈ ks1]

param_down = (nextfloat(0f0), g, δ₀, k_down, ħ, m)
ω₊_down = [dispersion_relation(k, param_down..., true) for k ∈ ks2]
ω₋_down = [dispersion_relation(k, param_down..., false) for k ∈ ks2]

with_theme(theme_latexfonts()) do
    fig = Figure(fontsize=20, size=(800, 400))
    ax1 = Axis(fig[1, 1]; xlabel=L"k", ylabel=L"\omega")
    ax2 = Axis(fig[1, 2]; xlabel=L"k")

    hideydecorations!(ax2, grid=false)
    ylims!(ax1, (-1, 1))
    ylims!(ax2, (-1, 1))

    lines!(ax1, ks1, ω₊_up, linewidth=4)
    lines!(ax1, ks1, ω₋_up, linewidth=4)
    lines!(ax2, ks2, ω₊_down, linewidth=4)
    lines!(ax2, ks2, ω₋_down, linewidth=4)
    fig
end
##
#Hawking (u1, d2)
param1 = (param_up..., true)
param2 = (param_down..., false)

k_ext1, ω_ext1 = find_extrema(dispersion_relation, (-1, 1), param1...)
k_ext2, ω_ext2 = find_extrema(dispersion_relation, (0.1, 1), param2...)

k1_min = find_zero(k -> dispersion_relation(k, param1...) - ω_ext2, (-1, k_ext1))
k2_min = find_zero(k -> dispersion_relation(k, param2...) - ω_ext1, (0, k_ext2))

bracket1 = (k1_min * 0.9999, k_ext1)
bracket2 = (k2_min, k_ext2)

corr_up_u1d2, corr_down_u1d2 = correlate(param1, bracket1, param2, bracket2, 128, false)
## (u, d1)
param1 = (param_up..., true)
param2 = (param_down..., true)

k_ext1, ω_ext1 = find_extrema(dispersion_relation, (-1, 1), param1...)

k1_min = find_zero(k -> dispersion_relation(k, param1...) - ω_ext2, (-1, k_ext1))

bracket1 = (-0.8, k_ext1)
bracket2 = (0.01, 1)

corr_up_u1d1, corr_down_u1d1 = correlate(param1, bracket1, param2, bracket2, 128, false)
##
ks = range(; start=-π / param.δL, step=2π / (size(G2_k, 1) * param.δL), length=size(G2_k, 1))
J = (N÷2-230:N÷2+230) .+ 90
power = 3

with_theme(theme_latexfonts()) do
    fig = Figure(; size=(850, 600), fontsize=20)
    ax = Axis(fig[1, 1], aspect=DataAspect(), xlabel=L"k", ylabel=L"k\prime")
    hm = heatmap!(ax, ks[J], ks[J], (G2_k[J, J] .- 1) * 10^3, colorrange=(-1, 1), colormap=:inferno)
    Colorbar(fig[1, 2], hm, label=L"g_2(k, k\prime) -1 \ \ ( \times 10^{-%$power})")
    for line_func! in (hlines!, vlines!)
        line_func!(ax, k_up, color=:green, linestyle=:dash)
        line_func!(ax, k_down, color=:green, linestyle=:dash)
    end
    lines!(ax, corr_down_u1d2 .+ k_down, corr_up_u1d2 .+ k_up, linewidth=4, color=:blue, linestyle=:dash, label="u1d2")
    lines!(ax, corr_down_u1d1 .+ k_down, corr_up_u1d1 .+ k_up, linewidth=4, color=:red, linestyle=:dash, label="u1d1")
    Legend(fig[1, 3], ax)
    fig
end