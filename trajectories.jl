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

    Ks = wavenumber(field, param.dx)
    cs = map((ψ, k, x) -> speed_of_sound(abs2(ψ), param.g, param.δ₀, k, param.ħ, param.m), steady_state, Ks, xs)
    μ²s = map((ψ, k, x) -> mass_term(abs2(ψ), param.g, param.δ₀, k, param.ħ, param.m), steady_state, Ks, xs)

    K_itp = linear_interpolation(xs .- param.x_def, Ks, extrapolation_bc=Flat())

    idxs = findall(!isnan, cs)
    c_itp = linear_interpolation(xs[idxs] .- param.x_def, cs[idxs], extrapolation_bc=Flat())

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
saving_dir = "/home/marcsgil/Code/LEON/MomentumCorrelations/full_sim2/"

steady_state, param = jldopen(joinpath(saving_dir, "steady_state.jld2")) do file
    file["steady_state"],
    file["param"]
end

K_itp, c_itp, μ²_itp = build_interpolations(steady_state, param, -100, 100)

xs = LinRange(-50, 50, 512)

lines(xs, c_itp(xs))
##
δω = 0.6

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
fig = Figure(size=(800, 600), fontsize=24)
ax = Axis(fig[1, 1], xlabel=L"x \ (\mu \text{m})", ylabel=L"t \ (\text{ps})")

f(x, p, t) = vg4(x)
u0 = x_turn * 0.999

tspan = (0.0, 3)
prob = ODEProblem(f, u0, tspan)
sol4 = solve(prob, Tsit5(), reltol=1e-8, abstol=1e-8)

lines!(ax, sol4.u, sol4.t, linewidth=2, color=:red, linestyle=:dash)
fig

f(x, p, t) = vg3(x)
u0 = sol4.u[end]

tspan = (-sol4.t[end], 0)
prob = ODEProblem(f, u0, tspan)
sol3 = solve(prob, Tsit5(), reltol=1e-8, abstol=1e-8)
lines!(ax, sol3.u, sol3.t, linewidth=2, color=:red)

f(x, p, t) = vg1(x)
u0 = x_turn * 0.999

tspan = (0, 3)
prob = ODEProblem(f, u0, tspan)
sol1 = solve(prob, Tsit5(), reltol=1e-8, abstol=1e-8)
lines!(ax, sol1.u, sol1.t, linewidth=2, color=:green)

tspan = (0, -3)
prob = ODEProblem(f, u0, tspan)
sol1 = solve(prob, Tsit5(), reltol=1e-8, abstol=1e-8)
lines!(ax, sol1.u, sol1.t, linewidth=2, color=:green)

f(x, p, t) = vg2(x)
u0 = x_turn * 0.999

tspan = (0, 3)
prob = ODEProblem(f, u0, tspan)
sol2 = solve(prob, Tsit5(), reltol=1e-8, abstol=1e-8)
lines!(ax, sol2.u, sol2.t, linewidth=2, color=:blue)

tspan = (0, -3)
prob = ODEProblem(f, u0, tspan)
sol2 = solve(prob, Tsit5(), reltol=1e-8, abstol=1e-8)
lines!(ax, sol2.u, sol2.t, linewidth=2, color=:blue)

xlims!(ax, 0, 6)

fig