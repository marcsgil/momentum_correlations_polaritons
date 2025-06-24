using CairoMakie, JLD2, MakieExtra
include("equations.jl")
include("io.jl")

saving_dir = "data"

steady_state, param = jldopen(joinpath(saving_dir, "steady_state.jld2")) do file
    file["steady_state"],
    file["param"]
end

window_pairs = read_window_pairs(saving_dir)
commutators = [calculate_momentum_commutators(first(pair).window, last(pair).window, first(pair).first_idx, last(pair).first_idx, param.dx) for pair in window_pairs]

xs = StepRangeLen(0, param.dx, param.N) .- param.x_def

momentum_averages = jldopen(joinpath(saving_dir, "averages.jld2")) do file
    [file["momentum_averages_$window_idx"] for window_idx in 1:length(window_pairs)]
end

g2_k = [fftshift(calculate_g2m1(momentum_average, commutator)) for (momentum_average, commutator) in zip(momentum_averages, commutators)]
ks = [fftshift(fftfreq(length(first(win).window), 2π / param.dx)) for win in window_pairs]

with_theme(theme_latexfonts()) do
    fig = Figure(; size=(900, 450), fontsize=14)

    g = fig.layout
    g_top = g[1, 1] = GridLayout()
    g_bottom = g[2, 1] = GridLayout()

    # Windows
    g_windows = g_top[1, 1] = GridLayout()
    ga = g_windows[1, 1] = GridLayout()
    gb = g_windows[1, 2] = GridLayout()
    axa = Axis(ga[1, 1], xlabel=L"x - x_H \ (\mu \text{m})", ylabel="(a. u.)")
    axb = Axis(gb[1, 1], xlabel=L"x - x_H \ (\mu \text{m})", ylabel="(a. u.)")

    for (ax, pair) in zip([axb, axa], window_pairs)
        window1 = first(pair).window
        window2 = last(pair).window
        first_idx1 = first(pair).first_idx
        first_idx2 = last(pair).first_idx

        lines!(ax, xs[first_idx1:first_idx1+length(window1)-1], window1, color=:blue, linewidth=4, linestyle=:dash)
        lines!(ax, xs[first_idx2:first_idx2+length(window2)-1], window2, color=:red, linewidth=4, linestyle=:dot)
    end

    #hidexdecorations!(axa; grid=false)

    linkxaxes!(axb, axa)

    # Correlations Large Windows
    pow = 3
    gc = g_bottom[1, 1] = GridLayout()

    axc = Axis(gc[1, 1]; aspect=DataAspect(),
        xlabel=L"k \ (\mu \text{m}^{-1})", ylabel=L"k\prime \ (\mu \text{m}^{-1})")
    hm = heatmap!(axc, ks[2], ks[2], g2_k[2] * 10^pow, colorrange=(-2, 2), colormap=:inferno)

    hlines!(axc, param.k_up, color=:deepskyblue, linestyle=:dashdot)
    vlines!(axc, param.k_down, color=:deepskyblue, linestyle=:dashdot)

    # Correlations Small Windows
    gd = g_bottom[1, 2] = GridLayout()
    axd = Axis(gd[1, 1]; aspect=DataAspect(),
        xlabel=L"k \ (\mu \text{m}^{-1})", ylabel=L"k\prime \ (\mu \text{m}^{-1})")
    hm = heatmap!(axd, ks[1], ks[1], g2_k[1] * 10^pow, colorrange=(-2, 2), colormap=:inferno)

    hlines!(axd, param.k_up, color=:deepskyblue, linestyle=:dashdot)
    vlines!(axd, param.k_down, color=:deepskyblue, linestyle=:dashdot)

    # Correlations small Windows zoomed
    ge = g_bottom[1, 3] = GridLayout()
    axe = Axis(ge[1, 1]; aspect=DataAspect(),
        xlabel=L"k \ (\mu \text{m}^{-1})", ylabel=L"k\prime \ (\mu \text{m}^{-1})")
    hm = heatmap!(axe, ks[1], ks[1], g2_k[1] * 10^pow, colorrange=(-2, 2), colormap=:inferno)
    xlims!(axe, param.k_down - 0.9, param.k_down + 0.9)
    ylims!(axe, param.k_up - 0.9, param.k_up + 0.9)

    hlines!(axe, param.k_up, color=:deepskyblue, linestyle=:dashdot)
    vlines!(axe, param.k_down, color=:deepskyblue, linestyle=:dashdot)

    space = :relative
    font = :bold
    color = :green1
    text!(axe, 0.75, 0.25; text="1", space, color, font)
    text!(axe, 0.75, 0.1; text="2", space, color, font)
    text!(axe, 0.2, 0.05; text="3", space, color, font)
    text!(axe, 0.1, 0.15; text="4", space, color, font)
    text!(axe, 0.15, 0.75; text="5", space, color, font)
    text!(axe, 0.25, 0.90; text="6", space, color, font)
    text!(axe, 0.71, 0.76; text="7", space, color, font)
    text!(axe, 0.69, 0.65; text="8", space, color, font)

    zoom_lines!(axd, axe)

    cb = Colorbar(fig[:, end+1], hm, label=L"g_2(k, k\prime) -1 \ \ ( \times 10^{-%$pow})")
    #cb.height = 840
    #cb.alignmode = Mixed(bottom=25)

    #rowsize!(g_windows, 1, Relative(0.5))

    #colsize!(g, 1, Relative(0.4))

    for (label, layout) in zip(["(a)", "(b)", "(c)", "(d)", "(e)"], [ga, gb, gc, gd, ge])
        Label(layout[1, 1, TopLeft()], label,
            fontsize=20,
            padding=(0, 40, 0, 0),
            halign=:right)
    end

    #rowgap!(g_windows, -10)
    #colsize!(g_top, 1, Relative(0.4))
    rowsize!(g, 1, Relative(0.25))
    rowgap!(g, 10)


    save(joinpath(saving_dir, "fig2.pdf"), fig)

    fig
end