using CairoMakie, HDF5, FFTW
include("../tracing.jl")
include("../io.jl")
include("../polariton_funcs.jl")
include("equations.jl")

saving_path = "PositiveP/correlations.h5"
group_name = "no_support"

param, steady_state, t_steady_state, one_point_r, two_point_r, one_point_k, two_point_k = h5open(saving_path) do file
    group = file[group_name]

    @info "Average over n_ave = $(group["n_ave"][1])"

    read_parameters(group),
    group["steady_state"] |> read,
    group["t_steady_state"] |> read,
    group["one_point_r"] |> read,
    group["two_point_r"] |> read,
    group["one_point_k"] |> read,
    group["two_point_k"] |> read
end

g2_r = calculate_g2(one_point_r, two_point_r)
g2_k = calculate_g2(one_point_k, two_point_k)

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
##
power = 5
with_theme(theme_latexfonts()) do
    fig = Figure(; size=(730, 600), fontsize=20)
    ax = Axis(fig[1, 1], aspect=DataAspect(), xlabel=L"x", ylabel=L"x\prime")
    xlims!(ax, (-150, 150))
    ylims!(ax, (-150, 150))
    hm = heatmap!(ax, rs, rs, (g2_r .- 1) * 10^power, colorrange=(-5,5), colormap=:inferno)
    Colorbar(fig[1, 2], hm, label=L"g_2(x, x\prime) -1 \ \ ( \times 10^{-%$power})")
    fig
end
##
ks = range(; start=-π / param.δL, step=2π / (size(g2_k, 1) * param.δL), length=size(g2_k, 1))
power = 3

ticks = [0.0]
ticklabels = [L"%$tick" for tick in ticks]
for (k, label) in zip((k_up, k_down, -k_down), (L"k_{\text{up}}", L"k_{\text{down}}", L"-k_{\text{down}}"))
    push!(ticks, k)
    push!(ticklabels, label)
end

with_theme(theme_latexfonts()) do
    fig = Figure(; size=(900, 600), fontsize=20)
    ax = Axis(fig[1, 1], aspect=DataAspect(), xlabel=L"k", ylabel=L"k\prime", xticks=(ticks, ticklabels), yticks=(ticks, ticklabels))
    hm = heatmap!(ax, ks, ks, (g2_k .- 1) * 10^3, colorrange=(-1, 1), colormap=:inferno)
    xlims!(ax, (-0.6, 1.2))
    ylims!(ax, (-0.6, 1.2))
    Colorbar(fig[1, 2], hm, label=L"g_2(k, k\prime) -1 \ \ ( \times 10^{-%$power})")
    for line_func! in (hlines!, vlines!)
        for k in (k_up, k_down, -k_down)
            line_func!(ax, k, color=:green, linestyle=:dash)
        end
    end

    #save("momentum_correlations.png", fig)
    fig
end