using JLD2, MakieExtra

include("plot_funcs.jl")
include("io.jl")
include("equations.jl")

saving_dir = "/home/marcsgil/Code/LEON/MomentumCorrelations/150_100um_window"

steady_state, param, = read_steady_state(saving_dir)

N = param.N
L = param.L
dx = param.dx
xs = StepRangeLen(0, dx, N) .- param.x_def
##
plot_bistability(xs, steady_state[1], param, -200, 200; saving_dir, factor_ns_down=1000, factor_ns_up=1.35)
##
ks_up = LinRange(-0.7, 0.7, 512)
ks_down = LinRange(-1.5, 1.5, 512)
plot_dispersion(xs, steady_state[1], param, -200, 200, 0.4, ks_up, ks_down; saving_dir)
##
plot_velocities(xs, steady_state[1], param; xlims=(-150, 150), ylims=(0, 3), saving_dir)
##
Vs = [potential((x + param.x_def,), param) for x in xs]

Is = [abs2(pump((x + param.x_def,), param, Inf)) for x in xs]

rVs = real.(Vs)
iVs = -imag.(Vs)

with_theme(theme_latexfonts()) do
    fig = Figure(; size=(700, 450), fontsize=20)
    ax1 = Axis(fig[1, 1]; ylabel="meV", xticks=-400:200:400)
    ax2 = Axis(fig[2, 1], xlabel=L"x \ (\mu \text{m})", ylabel=L" \hbar^3 g I \ (10^{-4} \ \text{meV}^3)", xticks=-400:200:400)
    hidexdecorations!(ax1; grid=false)
    ylims!(ax1, -1e-1, 1.1)
    ylims!(ax2, -1e-6, 4)
    #xlims!(ax1, -5, 500)
    #xlims!(ax2, -500, 500)

    ax_inset1 = Axis(fig[1, 2];
        yaxisposition=:right
    )
    hidexdecorations!(ax_inset1; grid=false)
    xlims!(ax_inset1, -20, 20)
    ylims!(ax_inset1, high=0.9)

    ax_inset2 = Axis(fig[2, 2];
        xlabel=L"x \ (\mu \text{m})",
        yaxisposition=:right)
    xlims!(ax_inset2, -20, 20)
    ylims!(ax_inset2, 0, 3)

    lines!(ax1, xs, param.ħ * rVs, color=:blue, linewidth=4)
    lines!(ax1, xs, param.ħ * iVs, color=:red, linewidth=4)
    lines!(ax2, xs, 10^4 * param.ħ^3 * param.g * Is, color=:green, linewidth=4)

    lines!(ax_inset1, xs, param.ħ * rVs, color=:blue, linewidth=4)
    lines!(ax_inset2, xs, 10^4 * param.ħ^3 * param.g * Is, color=:green, linewidth=4)

    zoom_lines!(ax1, ax_inset1)
    zoom_lines!(ax2, ax_inset2)

    fig
end