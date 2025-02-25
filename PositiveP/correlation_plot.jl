using CairoMakie, HDF5, FFTW
include("../tracing.jl")
include("../io.jl")
include("../polariton_funcs.jl")
include("equations.jl")

saving_path = "PositiveP/test.h5"
group_name = "PositiveP/test"

param, steady_state, t_steady_state, one_point_r, two_point_r, one_point_k, two_point_k = h5open(saving_path) do file
    group = file[group_name]

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

power = 5
with_theme(theme_latexfonts()) do
    fig = Figure(; size=(730, 600), fontsize=20)
    ax = Axis(fig[1, 1], aspect=DataAspect(), xlabel=L"x", ylabel=L"x\prime")
    xlims!(ax, (-100, 100))
    ylims!(ax, (-100, 100))
    hm = heatmap!(ax, rs, rs, (g2_r .- 1) * 10^power, colorrange=(-5,5), colormap=:inferno)
    Colorbar(fig[1, 2], hm, label=L"g_2(x, x\prime) -1 \ \ ( \times 10^{-%$power})")
    fig
end
##
power = 3
with_theme(theme_latexfonts()) do
    fig = Figure(; size=(730, 600), fontsize=20)
    ax = Axis(fig[1, 1], aspect=DataAspect(), xlabel=L"x", ylabel=L"x\prime")
    hm = heatmap!(ax, rs, rs, (g2_k .- 1) * 10^power, colorrange=(-5,5), colormap=:inferno)
    Colorbar(fig[1, 2], hm, label=L"g_2(x, x\prime) -1 \ \ ( \times 10^{-%$power})")
    fig
end