using CairoMakie, JLD2, Interpolations, Polynomials, DifferentialEquations, LaTeXStrings
include("polariton_funcs.jl")
include("equations.jl")

function build_interpolations(steady_state, param, xmin, xmax)
    xs = StepRangeLen(0, param.dx, param.N)
    idx_min = findfirst(x -> x - param.x_def >= xmin, xs)
    idx_max = findlast(x -> x - param.x_def <= xmax, xs)
    idxs = idx_min:idx_max
    xs = xs[idxs]
    steady_state = steady_state[1][idxs]

    Ks = wavenumber(steady_state, param.dx)
    cs = map((ψ, k, x) -> speed_of_sound(abs2(ψ), param.g, param.δ₀, k, param.ħ, param.m), steady_state, Ks, xs)
    μ²s = map((ψ, k, x) -> mass_term(abs2(ψ), param.g, param.δ₀, k, param.ħ, param.m), steady_state, Ks, xs)
    n = abs2.(steady_state)

    K_itp = linear_interpolation(xs .- param.x_def, Ks, extrapolation_bc=Flat())

    # Ignore NaNs in c
    idxs = findall(!isnan, cs)
    c_itp = linear_interpolation(xs[idxs] .- param.x_def, cs[idxs], extrapolation_bc=Flat())

    # Ignore NaNs in μ²
    # There is also a peak in the mass which we ignore
    μ²s[isnan.(μ²s)] .= -Inf
    μ²s[argmax(μ²s)] = -Inf
    μ²s[μ²s.<0] .= NaN
    idxs = findall(!isnan, μ²s)
    μ²_itp = linear_interpolation(xs[idxs] .- param.x_def, μ²s[idxs], extrapolation_bc=Flat())

    ns_itp = linear_interpolation(xs[idxs] .- param.x_def, n[idxs], extrapolation_bc=Flat())

    K_itp, c_itp, μ²_itp, ns_itp
end

function find_qs(δω, μ², c, K, ħ, m)
    v = ħ * K / m
    roots(Polynomial([μ² - δω^2, 2v * δω, c^2 - v^2, 0, (ħ / 2m)^2]))
end

bool2number(vals::Vararg{Bool,N}) where {N} = sum(n -> vals[n] * 2^(n - 1), 1:N)

function order_root(root, δω, vg, v)
    bool2number(vg > 0, δω < v * root) + 1
end

function symmetric_solution(f, u0, T, args...; kwargs...)
    prob1 = ODEProblem(f, u0, (0, T))
    sol1 = solve(prob1, args...; kwargs...)
    prob2 = ODEProblem(f, u0, (0, -T))
    sol2 = solve(prob2, args...; kwargs...)
    sol1, sol2
end

function get_trajectories(vg1, vg2, vg3, vg4, x_turn, T, args...; kwargs...)
    sol1 = symmetric_solution((x, p, t) -> vg1(x), x_turn, T, args...; kwargs...)
    sol2 = symmetric_solution((x, p, t) -> vg2(x), x_turn, T, args...; kwargs...)

    prob3 = ODEProblem((x, p, t) -> vg3(x), x_turn * (1 + sign(x_turn) * 0.01), (0, -T))
    sol3 = solve(prob3, args...; kwargs...)
    prob4 = ODEProblem((x, p, t) -> vg4(x), x_turn * (1 + sign(x_turn) * 0.01), (0, T))
    sol4 = solve(prob4, args...; kwargs...)

    sol1, sol2, sol3, sol4
end

function append_latexstring(string, substring)
    latexstring(string[begin:end-1] * substring * string[end])
end
##
saving_dir = "data"

steady_state, param = jldopen(joinpath(saving_dir, "steady_state.jld2")) do file
    file["steady_state"],
    file["param"]
end

position_averages = jldopen(joinpath(saving_dir, "averages.jld2")) do file
    n_ave = file["n_ave"][1]
    order_of_magnitude = round(Int, log10(n_ave))
    @info "Average after $(n_ave / 10^order_of_magnitude) × 10^$order_of_magnitude realizations"
    file["position_averages"]
end

commutators_r = calculate_position_commutators(param.N, param.dx)
g2_r = calculate_g2m1(position_averages, commutators_r)

N = param.N
L = param.L
dx = param.dx

g, δ₀, ħ, m, γ, F_up, F_down = param.g, param.δ₀, param.ħ, param.m, param.γ, param.F_up, param.F_down

δω = 0.4

K_itp, c_itp, μ²_itp, ns_itp = build_interpolations(steady_state, param, -200, 200)

with_theme(theme_latexfonts()) do
    fig = Figure(; size=(1600, 900), fontsize=26)
    linewidth = 4
    markersize = 14

    g_top = fig[1, 1] = GridLayout()
    g_bottom = fig[2, 1] = GridLayout()

    # Bistability
    ga = g_top[1, 1] = GridLayout()
    axa1 = Axis(ga[1, 1], title="Upstream", xlabel=L"\hbar^3 g I \times 10^3 \ (meV^3)", ylabel=L"\hbar gn \ (\text{meV})",)
    axa2 = Axis(ga[1, 2], title="Downstream", xlabel=L"\hbar^3 g I \times 10^3 \ (meV^3)")

    ns_up_theo = LinRange(0, 2050, 512)
    Is_up_theo = eq_of_state.(ns_up_theo, g, δ₀, K_itp(-Inf), ħ, m, γ)
    ns_down_theo = LinRange(0, 90, 512)
    Is_down_theo = eq_of_state.(ns_down_theo, g, δ₀, K_itp(Inf), ħ, m, γ)

    lines!(axa1, 10^3 * ħ^3 * g * Is_up_theo, ħ * g * ns_up_theo; color=:steelblue4, linewidth)
    lines!(axa2, 10^3 * ħ^3 * g * Is_down_theo, ħ * g * ns_down_theo; color=:steelblue4, linewidth)
    scatter!(axa1, 10^3 * ħ^3 * g * abs2.(F_up), ħ * g * ns_itp(-Inf); markersize=20, color=:black)
    scatter!(axa2, 10^3 * ħ^3 * g * abs2.(F_down), ħ * g * ns_itp(Inf); markersize=20, color=:black)

    # Dispersion relation
    gb = g_top[1, 2] = GridLayout()
    axb1 = Axis(gb[1, 1], xlabel=L"q \ (\mu \text{m}^{-1})", ylabel=L" \delta \omega \ (\text{ps}^{-1})", title="Upstream")
    axb2 = Axis(gb[1, 2], xlabel=L"q \ (\mu \text{m}^{-1})", title="Downstream")
    hideydecorations!(axb2; grid=false)
    linkyaxes!(axb1, axb2)

    ylims!(axb1, -1, 1)
    ylims!(axb2, -1, 1)

    qs_up = LinRange(-0.7, 0.7, 512)
    qs_down = LinRange(-1.5, 1.5, 512)

    δωs₊_up = dispersion_relation.(qs_up, μ²_itp(-Inf), c_itp(-Inf), K_itp(-Inf), ħ, m, true)
    δωs₋_up = dispersion_relation.(qs_up, μ²_itp(-Inf), c_itp(-Inf), K_itp(-Inf), ħ, m, false)
    δωs₊_down = dispersion_relation.(qs_down, μ²_itp(Inf), c_itp(Inf), K_itp(Inf), ħ, m, true)
    δωs₋_down = dispersion_relation.(qs_down, μ²_itp(Inf), c_itp(Inf), K_itp(Inf), ħ, m, false)
    lines!(axb1, qs_up, δωs₊_up; linewidth)
    lines!(axb1, qs_up, δωs₋_up; linewidth)
    lines!(axb2, qs_down, δωs₊_down; linewidth)
    lines!(axb2, qs_down, δωs₋_down; linewidth)

    for val ∈ (δω, -δω), ax ∈ (axb1, axb2)
        hlines!(ax, val; color=:black, linestyle=:dash, linewidth=2)
    end

    qs_up₊ = find_qs(δω, μ²_itp(-Inf), c_itp(-Inf), K_itp(-Inf), ħ, m)
    qs_up₊ = real.(qs_up₊[isreal.(qs_up₊)])
    qs_down₊ = real.(find_qs(δω, μ²_itp(Inf), c_itp(Inf), K_itp(Inf), ħ, m))

    scatter!(axb1, qs_up₊, fill(δω, 2); markersize, color=:black)
    scatter!(axb2, qs_down₊, fill(δω, 4); markersize, color=:black)

    qs_up₋ = find_qs(-δω, μ²_itp(-Inf), c_itp(-Inf), K_itp(-Inf), ħ, m)
    qs_up₋ = real.(qs_up₋[isreal.(qs_up₋)])
    qs_down₋ = real.(find_qs(-δω, μ²_itp(Inf), c_itp(Inf), K_itp(Inf), ħ, m))

    scatter!(axb1, qs_up₋, fill(-δω, 2); markersize, color=:black)
    scatter!(axb2, qs_down₋, fill(-δω, 4); markersize, color=:black)

    uin = "\$\\text{in}\$"
    uout = "\$\\text{HR}\$"
    d1in = "\$\\text{p}\$"
    d1out = "\$\\text{down}\$"
    d2in = "\$\\text{d}\$"
    d2out = "\$\\text{dn}\$"

    text!(axb1, qs_up₊[1], δω, text=latexstring(uout), fontsize=24, align=(:right, :top), offset=(10, -10))
    text!(axb1, qs_up₊[2], δω, text=latexstring(uin), fontsize=24, align=(:left, :top), offset=(10, 0))
    text!(axb1, qs_up₋[2], -δω, text=append_latexstring(uout, "^*"), fontsize=24, align=(:left, :bottom), offset=(5, 0))
    text!(axb1, qs_up₋[1], -δω, text=append_latexstring(uin, "^*"), fontsize=24, align=(:right, :bottom), offset=(-10, 0))

    text!(axb2, qs_down₊[2], δω, text=latexstring(d1out), fontsize=24, align=(:right, :bottom), offset=(-10, 0))
    text!(axb2, qs_down₊[1], δω, text=latexstring(d1in), fontsize=24, align=(:left, :bottom), offset=(10, 5))
    text!(axb2, qs_down₋[2], -δω, text=latexstring(append_latexstring(d2out, "^*")), fontsize=24, align=(:right, :bottom), offset=(-5, 5))
    text!(axb2, qs_down₋[1], -δω, text=latexstring(append_latexstring(d2in, "^*")), fontsize=24, align=(:right, :top), offset=(-10, -5))

    text!(axb2, qs_down₊[3], δω, text=latexstring(d2out), fontsize=24, align=(:left, :top), offset=(-5, -5))
    text!(axb2, qs_down₊[4], δω, text=latexstring(d2in), fontsize=24, align=(:left, :bottom), offset=(10, 0))
    text!(axb2, qs_down₋[3], -δω, text=latexstring(append_latexstring(d1out, "^*")), fontsize=24, align=(:left, :top), offset=(5, -5))
    text!(axb2, qs_down₋[4], -δω, text=latexstring(append_latexstring(d1in, "^*")), fontsize=24, align=(:right, :top), offset=(-10, -5))

    # Velocities
    gc = g_bottom[1, 1] = GridLayout()
    axc = Axis(gc[1, 1], xlabel=L"x \ (\mu \text{m})", ylabel=L"\mu \text{m} / \text{ps}")

    xs_vel = LinRange(-100, 100, 512)
    lines!(axc, xs_vel, ħ * K_itp(xs_vel) / m; linewidth, color=:orangered)
    lines!(axc, xs_vel, c_itp(xs_vel); linewidth, color=:deepskyblue)

    # Trajectories
    gd = g_bottom[1, 2] = GridLayout()
    axd = Axis(gd[1, 1], xlabel=L"x \ (\mu \text{m})", ylabel=L"t \ (\text{ps})")
    xs_traj = LinRange(-20, 20, 2^15)
    vgs = fill(NaN, 4, length(xs_traj))

    for n ∈ axes(vgs, 2)
        x = xs_traj[n]
        qs = find_qs(δω, μ²_itp(x), c_itp(x), K_itp(x), param.ħ, param.m)
        v = param.ħ * K_itp(x) / param.m

        for m ∈ axes(vgs, 1)
            if isreal(qs[m])
                q = real(qs[m])
                vg = group_velocity(q, δω, K_itp(x), c_itp(x), param.ħ, param.m)
                idx = order_root(q, δω, vg, v)
                vgs[idx, n] = vg
            end
        end
    end

    idx_turn = findlast(isnan, vgs[3, :])
    x_turn = xs_traj[idx_turn]

    vg1 = cubic_spline_interpolation(xs_traj, vgs[1, :], extrapolation_bc=Flat())
    vg2 = cubic_spline_interpolation(xs_traj, vgs[2, :], extrapolation_bc=Flat())
    vg3 = cubic_spline_interpolation(xs_traj[idx_turn:end], vcat(0, vgs[3, idx_turn+1:end]), extrapolation_bc=Flat())
    vg4 = cubic_spline_interpolation(xs_traj[idx_turn:end], vcat(0, vgs[4, idx_turn+1:end]), extrapolation_bc=Flat())

    sol1, sol2, sol3, sol4 = get_trajectories(vg1, vg2, vg3, vg4, x_turn, 3, Tsit5(), reltol=1e-8, abstol=1e-8)
    lines!(axd, sol1[1].u, sol1[1].t; linewidth, color=:green)
    lines!(axd, sol1[2].u, sol1[2].t; linewidth, color=:green)
    lines!(axd, sol2[1].u, sol2[1].t; linewidth, color=:blue)
    lines!(axd, sol2[2].u, sol2[2].t; linewidth, color=:blue)
    lines!(axd, sol3.u, sol3.t; linewidth, color=:red)
    lines!(axd, sol4.u, sol4.t; linewidth, color=:red, linestyle=:dash)

    text!(axd, sol1[1].u[end], sol1[1].t[end]; text=latexstring(uout), align=(:right, :bottom))
    text!(axd, sol1[2].u[end], sol1[2].t[end]; text=latexstring(d1in), align=(:left, :top))
    text!(axd, sol2[1].u[end], sol2[1].t[end]; text=latexstring(d1out), align=(:left, :bottom), offset=(-10, 0))
    text!(axd, sol2[2].u[end], sol2[2].t[end]; text=latexstring(uin), align=(:left, :top))
    text!(axd, sol3.u[end], sol3.t[end]; text=latexstring(d2in), align=(:left, :top))
    text!(axd, sol4.u[end], sol4.t[end]; text=latexstring(d2out), align=(:left, :bottom))

    xlims!(axd, -6, 12)
    ylims!(axd, -4, 4)

    # Position correlations
    xs_corr = StepRangeLen(0, param.dx, param.N) .- param.x_def
    ge = g_bottom[1, 3] = GridLayout()
    pow = 5
    axe = Axis(ge[1, 1], aspect=DataAspect(), xlabel=L"x \ (\mu \text{m})", ylabel=L"x\prime \ (\mu \text{m})")
    xlims!(axe, -140, 140)
    ylims!(axe, -140, 140)
    hm = heatmap!(axe, xs_corr, xs_corr, g2_r * 10^pow, colorrange=(-6, 6), colormap=:inferno)
    Colorbar(ge[1, 2], hm, label=L"g_2(x, x\prime) -1 \ \ ( \times 10^{-%$pow})")
    fig

    for (label, layout) in zip(["(A)", "(B)", "(C)", "(D)", "(E)"], [ga, gb, gc, gd, ge])
        Label(layout[1, 1, TopLeft()], label,
            fontsize=26,
            padding=(0, 50, 20, 0),
            halign=:right)
    end

    colsize!(g_top, 1, 500)

    save(joinpath(saving_dir, "fig1.pdf"), fig)

    fig
end