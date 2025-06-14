using CairoMakie
include("tracing.jl")
include("io.jl")

function plot_density!(ax, rs, field, param; xlims=nothing, ylims=nothing)
    n = Array(abs2.(field))
    lines!(ax, rs, param.g * n, linewidth=4)
    !isnothing(xlims) && xlims!(ax, xlims...)
    !isnothing(ylims) && ylims!(ax, ylims...)
end

function plot_density(rs, field, param; xlims=nothing, ylims=nothing, fontsize=24, saving_dir=nothing)
    with_theme(theme_latexfonts()) do
        fig = Figure(; fontsize)
        ax = Axis(fig[1, 1], xlabel=L"x", ylabel=L"gn")
        plot_density!(ax, rs, field, param; xlims, ylims)
        isnothing(saving_dir) || save(joinpath(saving_dir, "density.pdf"), fig)
        fig
    end
end

function plot_velocities!(ax, rs, field, param; xlims=nothing, ylims=nothing, show_legend=true)
    v = velocity(Array(field), param.ħ, param.m, param.dx)

    ks = wavenumber(field, param.dx)
    c = map((ψ, k) -> speed_of_sound(abs2(ψ), param.g, param.δ₀, k, param.ħ, param.m), Array(field), ks)

    !isnothing(xlims) && xlims!(ax, xlims...)
    !isnothing(ylims) && ylims!(ax, ylims...)
    lines!(ax, rs, c, linewidth=4, color=:blue, label=L"c")
    lines!(ax, rs, v, linewidth=4, color=:red, label=L"v")
    show_legend && axislegend(; position=:lt)
end

function plot_velocities(rs, field, param; xlims=nothing, ylims=nothing, fontsize=24, saving_dir=nothing)
    with_theme(theme_latexfonts()) do
        fig = Figure(; fontsize)
        ax = Axis(fig[1, 1], xlabel=L"x \ (\mu \text{m})", ylabel=L"\mu \text{m} / \text{ps}")
        plot_velocities!(ax, rs, field, param; xlims, ylims)
        isnothing(saving_dir) || save(joinpath(saving_dir, "velocities.pdf"), fig)
        fig
    end
end

function plot_bistability!(ax1, ax2, rs, steady_state, param, x_up, x_down, factor_ns_up=1.2, factor_ns_down=3)
    idx_up = argmin(idx -> abs(rs[idx] - x_up), eachindex(rs))
    idx_down = argmin(idx -> abs(rs[idx] - x_down), eachindex(rs))
    n_up = abs2(Array(steady_state)[idx_up])
    n_down = abs2(Array(steady_state)[idx_down])

    δ_up = param.δ₀ - param.ħ * param.k_up^2 / (2 * param.m)
    δ_down = param.δ₀ - param.ħ * param.k_down^2 / (2 * param.m)

    ns_up_theo = LinRange(0, factor_ns_up * δ_up / param.g, 512)
    Is_up_theo = eq_of_state.(ns_up_theo, param.g, param.δ₀, param.k_up, param.ħ, param.m, param.γ)
    ns_down_theo = LinRange(0, factor_ns_down * δ_down / param.g, 512)
    Is_down_theo = eq_of_state.(ns_down_theo, param.g, param.δ₀, param.k_down, param.ħ, param.m, param.γ)

    lines!(ax1, 10^3 * param.ħ^3 * param.g * Is_up_theo, param.ħ * param.g * ns_up_theo, color=:blue, linewidth=4)
    lines!(ax2, 10^3 * param.ħ^3 * param.g * Is_down_theo, param.ħ * param.g * ns_down_theo, color=:red, linewidth=4)
    scatter!(ax1, 10^3 * param.ħ^3 * param.g * abs2(param.F_up), param.ħ * param.g * n_up, color=:black, markersize=16)
    scatter!(ax2, 10^3 * param.ħ^3 * param.g * abs2(param.F_down), param.ħ * param.g * n_down, color=:black, markersize=16)
end

function plot_bistability(rs, steady_state, param, x_up, x_down; factor_ns_up=1.2, factor_ns_down=3, saving_dir=nothing)
    with_theme(theme_latexfonts()) do
        fig = Figure(fontsize=24)
        ax1 = Axis(fig[1, 1]; xlabel=L"\hbar^3 g I \times 10^3 \ (meV^3)", ylabel=L"\hbar g n \ (meV)", title="Upstream")
        ax2 = Axis(fig[1, 2]; xlabel=L"\hbar^3 g I \times 10^3 \ (meV^3)", ylabel=L"\hbar g n \ (meV)", title="Downstream")
        plot_bistability!(ax1, ax2, rs, steady_state, param, x_up, x_down, factor_ns_up, factor_ns_down)
        isnothing(saving_dir) || save(joinpath(saving_dir, "bistability.pdf"), fig)
        fig
    end
end

function plot_dispersion!(ax1, ax2, rs, steady_state, param, x_up, x_down, Ω, ks_up, ks_down)
    idx_up = argmin(idx -> abs(rs[idx] - x_up), eachindex(rs))
    idx_down = argmin(idx -> abs(rs[idx] - x_down), eachindex(rs))
    n_up = abs2(Array(steady_state)[idx_up])
    n_down = abs2(Array(steady_state)[idx_down])

    param_up = (n_up, param.g, param.δ₀, param.k_up, param.ħ, param.m)
    ω₊_up = [dispersion_relation(k, param_up..., true) for k ∈ ks_up]
    ω₋_up = [dispersion_relation(k, param_up..., false) for k ∈ ks_up]

    param_down = (n_down, param.g, param.δ₀, param.k_down, param.ħ, param.m)
    ω₊_down = [dispersion_relation(k, param_down..., true) for k ∈ ks_down]
    ω₋_down = [dispersion_relation(k, param_down..., false) for k ∈ ks_down]

    k_up_out = find_zero(k -> dispersion_relation(k, param_up..., true) - Ω, (-1, 0))
    k_up_in = find_zero(k -> dispersion_relation(k, param_up..., true) - Ω, (0, 1))

    k_star_up_out = find_zero(k -> dispersion_relation(k, param_up..., false) + Ω, (0, 1))
    k_star_up_in = find_zero(k -> dispersion_relation(k, param_up..., false) + Ω, (-1, 0))

    k1_down_out = find_zero(k -> dispersion_relation(k, param_down..., true) - Ω, (0.01, 1))
    k1_down_in = find_zero(k -> dispersion_relation(k, param_down..., true) - Ω, (-2, -0.01))

    k2_star_down_out = find_zero(k -> dispersion_relation(k, param_down..., true) + Ω, (-0.6, -0.01))
    k2_star_down_in = find_zero(k -> dispersion_relation(k, param_down..., true) + Ω, (-2, -0.6))

    k2_down_out = find_zero(k -> dispersion_relation(k, param_down..., false) - Ω, (0, 0.6))
    k2_down_in = find_zero(k -> dispersion_relation(k, param_down..., false) - Ω, (0.6, 2))

    k1_star_down_out = find_zero(k -> dispersion_relation(k, param_down..., false) + Ω, (-0.6, 0))
    k1_star_down_in = find_zero(k -> dispersion_relation(k, param_down..., false) + Ω, (0.6, 2))

    lines!(ax1, ks_up, ω₊_up, linewidth=4)
    lines!(ax1, ks_up, ω₋_up, linewidth=4)
    lines!(ax2, ks_down, ω₊_down, linewidth=4)
    lines!(ax2, ks_down, ω₋_down, linewidth=4)

    for ax ∈ (ax1, ax2)
        hlines!(ax, [Ω, -Ω], linestyle=:dash, color=:black)
    end

    scatter!(ax1, k_up_out, Ω, color=:black, markersize=16)
    text!(ax1, k_up_out, Ω, text=L"u_\text{out}", fontsize=24, align=(:right, :top), offset=(10, 0))
    scatter!(ax1, k_up_in, Ω, color=:black, markersize=16)
    text!(ax1, k_up_in, Ω, text=L"u_\text{in}", fontsize=24, align=(:left, :top), offset=(10, 0))
    scatter!(ax1, k_star_up_out, -Ω, color=:black, markersize=16)
    text!(ax1, k_star_up_out, -Ω, text=L"u_\text{out}^*", fontsize=24, align=(:left, :bottom), offset=(5, 0))
    scatter!(ax1, k_star_up_in, -Ω, color=:black, markersize=16)
    text!(ax1, k_star_up_in, -Ω, text=L"u_\text{in}^*", fontsize=24, align=(:right, :bottom), offset=(-10, 0))

    scatter!(ax2, k1_down_out, Ω, color=:red, markersize=16)
    text!(ax2, k1_down_out, Ω, text=L"d1_\text{out}", fontsize=24, align=(:right, :bottom), offset=(-10, 0), color=:red)
    scatter!(ax2, k1_down_in, Ω, color=:red, markersize=16)
    text!(ax2, k1_down_in, Ω, text=L"d1_\text{in}", fontsize=24, align=(:left, :bottom), offset=(10, 0), color=:red)
    scatter!(ax2, k2_star_down_out, -Ω, color=:red, markersize=16)
    text!(ax2, k2_star_down_out, -Ω, text=L"d2_\text{out}^*", fontsize=24, align=(:right, :bottom), offset=(-5, 5), color=:red)
    scatter!(ax2, k2_star_down_in, -Ω, color=:red, markersize=16)
    text!(ax2, k2_star_down_in, -Ω, text=L"d2_\text{in}^*", fontsize=24, align=(:right, :top), offset=(-10, -5), color=:red)

    scatter!(ax2, k2_down_out, Ω, color=:green, markersize=16)
    text!(ax2, k2_down_out, Ω, text=L"d2_\text{out}", fontsize=24, align=(:left, :top), offset=(-5, -5), color=:green)
    scatter!(ax2, k2_down_in, Ω, color=:green, markersize=16)
    text!(ax2, k2_down_in, Ω, text=L"d2_\text{in}", fontsize=24, align=(:left, :bottom), offset=(10, 0), color=:green)
    scatter!(ax2, k1_star_down_out, -Ω, color=:green, markersize=16)
    text!(ax2, k1_star_down_out, -Ω, text=L"d1_\text{out}^*", fontsize=24, align=(:left, :top), offset=(5, -5), color=:green)
    scatter!(ax2, k1_star_down_in, -Ω, color=:green, markersize=16)
    text!(ax2, k1_star_down_in, -Ω, text=L"d1_\text{in}^*", fontsize=24, align=(:right, :top), offset=(-10, -5), color=:green)
end

function plot_dispersion(rs, steady_state, param, x_up, x_down, Ω, ks_up, ks_down; saving_dir=nothing)
    with_theme(theme_latexfonts()) do
        fig = Figure(fontsize=24, size=(800, 450))
        ax1 = Axis(fig[1, 1]; xlabel=L"q \ (\mu \text{m}^{-1})", ylabel=L"\delta \omega \ (ps^{-1})", title="Upstream")
        ax2 = Axis(fig[1, 2]; xlabel=L"q \ (\mu \text{m}^{-1})", title="Downstream")

        hideydecorations!(ax2, grid=false)
        ylims!(ax1, (-1, 1))
        ylims!(ax2, (-1, 1))
        plot_dispersion!(ax1, ax2, rs, steady_state, param, x_up, x_down, Ω, ks_up, ks_down)
        isnothing(saving_dir) || save(joinpath(saving_dir, "dispersion.pdf"), fig)
        fig
    end
end

function plot_window_pair(saving_dir, n, pair, steady_state, rs, param; xlims=nothing, ylims=nothing, savefig=false)
    with_theme(theme_latexfonts()) do
        fig = Figure(fontsize=24, size=(550,450))
        ax = Axis(fig[1, 1], xlabel=L"x \ (\mu \text{m})")
        isnothing(xlims) || xlims!(ax, xlims...)
        isnothing(ylims) || ylims!(ax, ylims...)

        window1 = pair.first.window
        window2 = pair.second.window
        first_idx1 = pair.first.first_idx
        first_idx2 = pair.second.first_idx
        N1 = length(window1)
        N2 = length(window2)
        linewidth = 4

        lines!(ax, rs[first_idx1:first_idx1+N1-1], window1 ; linewidth, label=L"k \text{ Window}", linestyle=:dash)
        lines!(ax, rs[first_idx2:first_idx2+N2-1], window2; linewidth, label=L"k^\prime \text{ Window}", linestyle=:dot)
        #plot_velocities!(ax, rs, steady_state[1], param; xlims, ylims, show_legend=false)
        #Legend(fig[1, 2], ax)
        axislegend(ax)
        display(fig)

        savefig && save(joinpath(saving_dir, "window_pair_$n.pdf"), fig)
    end
end

function plot_all_windows(saving_dir; kwargs...)
    window_pairs = read_window_pairs(saving_dir)
    steady_state, param = read_steady_state(saving_dir)
    rs = StepRangeLen(0, param.dx, param.N) .- param.x_def
    for (n, pair) in enumerate(window_pairs)
        plot_window_pair(saving_dir, n, pair, steady_state, rs, param; kwargs...)
    end
    nothing
end