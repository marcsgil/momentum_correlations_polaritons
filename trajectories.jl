using CairoMakie, JLD2, Interpolations, Polynomials, DifferentialEquations
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

    K_itp, c_itp, μ²_itp
end

function find_qs(δω, μ², c, K, ħ, m)
    v = ħ * K / m
    roots(Polynomial([μ² - δω^2, 2v * δω, c^2 - v^2, 0, (ħ / 2m)^2]))
end

bool2number(vals::Vararg{Bool,N}) where {N} = sum(n -> vals[n] * 2^(n - 1), 1:N)
##
saving_dir = "data"

steady_state, param = jldopen(joinpath(saving_dir, "steady_state.jld2")) do file
    file["steady_state"],
    file["param"]
end

K_itp, c_itp, μ²_itp = build_interpolations(steady_state, param, -200, 200)
xs = LinRange(-200, 200, 2^10)

lines(xs, K_itp(xs))
##
δω = 0.4

xs = LinRange(-20, 20, 2^15)
vgs = Array{Float64}(undef, 4, length(xs))
vgs .= NaN

for n ∈ axes(vgs, 2)
    x = xs[n]
    qs = find_qs(δω, μ²_itp(x), c_itp(x), K_itp(x), param.ħ, param.m)
    v = param.ħ * K_itp(x) / param.m

    for m ∈ axes(vgs, 1)
        if isreal(qs[m])
            q = real(qs[m])
            vg = group_velocity(q, δω, K_itp(x), c_itp(x), param.ħ, param.m)
            idx = bool2number(vg > 0, δω < v * q) + 1
            vgs[idx, n] = vg
        end
    end
end

fig = Figure(size=(800, 600), fontsize=24)
ax = Axis(fig[1, 1], xlabel=L"x \ (\mu \text{m})", ylabel=L"v_g \ (\mu \text{m/ ps})")
series!(ax, xs, vgs, linewidth=2)
axislegend()
fig
##
idx_turn = findlast(isnan, vgs[3, :])
x_turn = xs[idx_turn]

vg1 = cubic_spline_interpolation(xs, vgs[1, :], extrapolation_bc=Flat())
vg2 = cubic_spline_interpolation(xs, vgs[2, :], extrapolation_bc=Flat())
vg3 = cubic_spline_interpolation(xs[idx_turn:end], vcat(0, vgs[3, idx_turn+1:end]), extrapolation_bc=Flat())
vg4 = cubic_spline_interpolation(xs[idx_turn:end], vcat(0, vgs[4, idx_turn+1:end]), extrapolation_bc=Flat())
##
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

sol1, sol2, sol3, sol4 = get_trajectories(vg1, vg2, vg3, vg4, x_turn, 3, Tsit5(), reltol=1e-8, abstol=1e-8)


with_theme(theme_latexfonts()) do
    fig = Figure(size=(800, 600), fontsize=24)
    ax = Axis(fig[1, 1], xlabel=L"x \ (\mu \text{m})", ylabel=L"t \ (\text{ps})", title = L"Scattering at $\delta\omega = %$δω$ ps$^{-1}$")
    lines!(ax, sol1[1].u, sol1[1].t, linewidth=2, color=:green)
    lines!(ax, sol1[2].u, sol1[2].t, linewidth=2, color=:green)
    lines!(ax, sol2[1].u, sol2[1].t, linewidth=2, color=:blue)
    lines!(ax, sol2[2].u, sol2[2].t, linewidth=2, color=:blue)
    lines!(ax, sol3.u, sol3.t, linewidth=2, color=:red)
    lines!(ax, sol4.u, sol4.t, linewidth=2, color=:red, linestyle=:dash)
    #save(joinpath(saving_dir, "trajectories_δω_$(δω).png"), fig)
    fig
end
