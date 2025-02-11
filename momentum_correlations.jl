using CairoMakie, HDF5
include("tracing.jl")

G2_r, G2_k = h5open("g2_momentum.h5", "r") do file
    read(file, "G2_r"), read(file, "G2_k")
end

L = 800.0f0
N = 1024
δL = L / N
k_up = 0.25f0
#g = 0.0003f0 / ħ
#n₀ = abs2(Array(steady_state)[N÷4])
k_down = √(2m * δ₀ / ħ)
##
ks = LinRange(-1, 1, 512)

param_up = (k_up, g, n₀, δ, m)
ω₊_up = dispersion_relation.(ks, param_up..., true)
ω₋_up = dispersion_relation.(ks, param_up..., false)

param_down = (k_down, g, 0, 0, m)
ω₊_down = dispersion_relation.(ks, param_down..., true)
ω₋_down = dispersion_relation.(ks, param_down..., false)

with_theme(theme_latexfonts()) do
    fig = Figure(fontsize=20, size=(800, 400))
    ax1 = Axis(fig[1, 1]; xlabel=L"k", ylabel=L"\omega", yticks=-1:0.1:1)
    ax2 = Axis(fig[1, 2]; xlabel=L"k", yticks=-1:0.1:1)

    ylims!(ax1, -0.9, 0.9)
    ylims!(ax2, -0.9, 0.9)

    hideydecorations!(ax2, grid=false)

    lines!(ax1, ks, ω₊_up, linewidth=4)
    lines!(ax1, ks, ω₋_up, linewidth=4)
    lines!(ax2, ks, ω₊_down, linewidth=4)
    lines!(ax2, ks, ω₋_down, linewidth=4)
    fig
end
##
up_bracket = -1, 0
down_bracket = 0.01, 0.7

k_ext_up, ω_ext_up = find_extrema(dispersion_relation, up_bracket, param_up..., true)
k_ext_down, ω_ext_down = find_extrema(dispersion_relation, down_bracket, param_down..., false)

ωs = LinRange(minmax(ω_ext_up, ω_ext_down)..., 512)

half_up_bracket = (-1, k_ext_up)
half_down_bracket = (0.01, k_ext_down)

corr_up_hw = [find_zero(k -> dispersion_relation(k, param_up..., true) - ω, half_up_bracket, Bisection()) for ω ∈ ωs]
corr_down_hw = [find_zero(k -> dispersion_relation(k, param_down..., false) - ω, half_down_bracket, Bisection()) for ω ∈ ωs]

lines(corr_up_hw, corr_down_hw, linewidth=4)
##
up_bracket = -0.5, 0

k_ext_up, ω_ext_up = find_extrema(dispersion_relation, up_bracket, param_up..., true)

ωs = LinRange(minmax(ω_ext_up, 1)..., 512)

half_up_bracket = (-1, k_ext_up)
half_down_bracket = (0, 1)

corr_up = [find_zero(k -> dispersion_relation(k, param_up..., true) - ω, half_up_bracket, Bisection()) for ω ∈ ωs]
corr_down = [find_zero(k -> dispersion_relation(k, param_down..., true) - ω, k_down, Order1()) for ω ∈ ωs]

lines(corr_up, corr_down, linewidth=4)
##
J = N÷2-120:N÷2+120 .+ 50
ks = range(; start=-π / δL, step=2π / (N * δL), length=N)
rs = range(; start=-L / 2, step=δL, length=N)

with_theme(theme_latexfonts()) do
    fig = Figure(; size=(730, 600), fontsize=20)
    ax = Axis(fig[1, 1], aspect=DataAspect(), xlabel=L"k", ylabel=L"k\prime")
    #hm = heatmap!(ax, rs[J], rs[J], (Array(real(G2_r)[J, J]) .- 1) * 1e5, colorrange=(-5, 5), colormap=:inferno)
    hm = heatmap!(ax, ks[J], ks[J], (G2_k[J, J] .- 1) * 1e3, colorrange=(-2, 2), colormap=:inferno)
    Colorbar(fig[1, 2], hm, label=L"g_2(k, k\prime) -1 \ \ ( \times 10^{-3})")
    for line_func! in (hlines!, vlines!)
        line_func!(ax, k_pump, color=:green, linestyle=:dash)
        line_func!(ax, k_down, color=:green, linestyle=:dash)
    end
    #lines!(ax, corr_down .+ k_down, corr_up .+ k_pump, linewidth=4, color=:blue, linestyle=:dot)
    #lines!(ax, corr_up .+ k_pump, corr_down .+ k_down, linewidth=4, color=:blue, linestyle=:dot)
    #lines!(ax, corr_down_hw .+ k_down, corr_up_hw .+ k_pump, linewidth=4, color=:purple, linestyle=:dot)
    #lines!(ax, corr_up_hw .+ k_pump, corr_down_hw .+ k_down, linewidth=4, color=:purple, linestyle=:dot)
    #save("dev_env/g2m1.pdf", fig)
    fig
end