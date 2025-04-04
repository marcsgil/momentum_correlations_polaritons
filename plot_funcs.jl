include("tracing.jl")

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

function plot_bistability!(ax1, ax2, rs, steady_state, param, x_up, x_down)
    idx_up = argmin(idx->abs(rs[idx] - x_up), eachindex(rs))
    idx_down = argmin(idx->abs(rs[idx] - x_down), eachindex(rs))
    n_up = abs2(Array(steady_state)[idx_up])
    n_down = abs2(Array(steady_state)[idx_down])

    δ_up = δ₀ - param.ħ * param.k_up^2 / (2 * param.m)
    δ_down = δ₀ - param.ħ * param.k_down^2 / (2 * param.m)

    ns_up_theo = LinRange(0, 1.2 * δ_up / g, 512)
    Is_up_theo = eq_of_state.(ns_up_theo, param.g, param.δ₀, param.k_up, param.ħ, param.m, param.γ)
    ns_down_theo = LinRange(0, 2 * δ_down / g, 512)
    Is_down_theo = eq_of_state.(ns_down_theo, param.g, param.δ₀, param.k_down, param.ħ, param.m, param.γ)

    lines!(ax1, Is_up_theo, ns_up_theo, color=:blue, linewidth=4)
    lines!(ax2, Is_down_theo, ns_down_theo, color=:red, linewidth=4)
    scatter!(ax1, abs2(param.F_up), n_up, color=:black, markersize=16)
    scatter!(ax2, abs2(param.F_down), n_down, color=:black, markersize=16)
end

function plot_bistability(rs, steady_state, param, x_up, x_down)
    with_theme(theme_latexfonts()) do
        fig = Figure(fontsize=16)
        ax1 = Axis(fig[1, 1]; xlabel="I", ylabel="n", title="Upstream")
        ax2 = Axis(fig[1, 2]; xlabel="I", ylabel="n", title="Downstream")
        plot_bistability!(ax1, ax2, rs, steady_state, param, x_up, x_down)
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
    text!(ax1, k_up_out, Ω, text=L"u_\text{out}", fontsize=24, align=(:right, :bottom), offset=(-10, 0))
    scatter!(ax1, k_up_in, Ω, color=:black, markersize=16)
    text!(ax1, k_up_in, Ω, text=L"u_\text{in}", fontsize=24, align=(:right, :bottom), offset=(-10, 0))
    scatter!(ax1, k_star_up_out, -Ω, color=:black, markersize=16)
    text!(ax1, k_star_up_out, -Ω, text=L"u_\text{out}^*", fontsize=24, align=(:right, :bottom), offset=(-10, 0))
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

function plot_dispersion(rs, steady_state, param, x_up, x_down, Ω, ks_up, ks_down)
    with_theme(theme_latexfonts()) do
        fig = Figure(fontsize=20, size=(800, 400))
        ax1 = Axis(fig[1, 1]; xlabel=L"\delta k", ylabel=L"\delta \omega", title="Upstream")
        ax2 = Axis(fig[1, 2]; xlabel=L"\delta k", title="Downstream")
    
        hideydecorations!(ax2, grid=false)
        ylims!(ax1, (-1, 1))
        ylims!(ax2, (-1, 1))
        plot_dispersion!(ax1, ax2, rs, steady_state, param, x_up, x_down, Ω, ks_up, ks_down)
        fig
    end
end