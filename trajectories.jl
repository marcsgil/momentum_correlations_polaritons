using CairoMakie, JLD2, Interpolations, Polynomials, LsqFit, DifferentialEquations
include("polariton_funcs.jl")

function model(x, p)
    y₋, y₊, x₀, k = p
    @. tanh(k * (x - x₀)) * (y₊ - y₋) / 2 + (y₊ + y₋) / 2
end

function get_fitted_model(xs, ys, y₋, y₊, p0)
    small_model(x, p) = model(x, (y₋, y₊, p[1], p[2]))
    fit = curve_fit(small_model, xs, ys, p0)
    x -> model(x, (y₋, y₊, coef(fit)[1], coef(fit)[2]))
end

function build_interpolations(steady_state, param, xmin, xmax)
    xs = StepRangeLen(0, param.dx, param.N - 1) .- param.x_def
    ns = abs2.(steady_state[1][2:end])
    ks = diff(unwrap(angle.(steady_state[1]))) / param.dx
    cs = map((n, k) -> speed_of_sound(n, param.g, param.δ₀, k, param.ħ, param.m), ns, ks)

    idx_min = findfirst(x -> x >= xmin, xs)
    idx_max = findlast(x -> x <= xmax, xs)
    idxs = idx_min:idx_max

    p0 = [-7.0, 10.0]

    n_itp = get_fitted_model(xs[idxs], ns[idxs], ns[idx_min], ns[idx_max], p0)
    k_itp = get_fitted_model(xs[idxs], ks[idxs], ks[idx_min], ks[idx_max], p0)

    idxs_c = findall(!isnan, cs[idxs])
    c_itp = get_fitted_model(xs[idxs][idxs_c], cs[idxs][idxs_c], cs[idx_min], cs[idx_max], p0)

    n_itp, k_itp, c_itp
end

function find_qs(δω, n, K, c, param)
    δ = detuning(param.δ₀, K, param.ħ, param.m)
    v = param.ħ * K / param.m
    mass_term = (3param.g * n - δ) * (param.g * n - δ)
    roots(Polynomial([mass_term - δω^2, 2v * δω, c^2 - v^2, 0, (param.ħ / 2 / param.m)^2]))
end

function group_velocity(q, δω, K, c, param)
    v = param.ħ * K / param.m
    v + q * (c^2 + param.ħ^2 * q^2 / 2 / param.m^2) / (δω - v * q)
end
##
saving_dir = "/home/marcsgil/Code/LEON/MomentumCorrelations/full_sim2/"

steady_state, param, t_steady_state = jldopen(joinpath(saving_dir, "steady_state.jld2")) do file
    file["steady_state"],
    file["param"],
    file["t_steady_state"]
end

n_itp, k_itp, c_itp = build_interpolations(steady_state, param, -30, 30)

xs_sim = (StepRangeLen(0, param.dx, param.N - 1) .- param.x_def)
ns = abs2.(steady_state[1][2:end])
ks = diff(unwrap(angle.(steady_state[1]))) / param.dx
cs = map((n, k) -> speed_of_sound(n, param.g, param.δ₀, k, param.ħ, param.m), ns, ks)

xs = LinRange(-50, 50, 1024)

fig = Figure(size=(800, 800), fontsize=24)
axs = [Axis(fig[n, 1], xlabel=L"x \ (\mu \text{m})") for n ∈ 1:3]

for (func, data, ax) ∈ zip(
    [n_itp, k_itp, c_itp],
    [ns, ks, cs],
    axs
)
    scatter!(ax, xs_sim, data, color=:black, label="Simulation")
    lines!(ax, xs, func.(xs), color=:red, linewidth=2, label="Interpolation")
    xlims!(ax, extrema(xs))
end
fig
##
δω = 0.65

xs = LinRange(-10, 10, 2^16)
vs = Array{Float64}(undef, 4, length(xs))
vs .= NaN

bool2number(vals::Vararg{Bool,N}) where {N} = sum(n -> vals[n] * 2^(n - 1), 1:N)

for n ∈ axes(vs, 2)
    x = xs[n]
    qs = find_qs(δω, n_itp(x), k_itp(x), c_itp(x), param)
    v = param.ħ * k_itp(x) / param.m

    for m ∈ axes(vs, 1)
        if isreal(qs[m])
            q = real(qs[m])
            v = group_velocity(q, δω, k_itp(x), c_itp(x), param)
            idx = bool2number(v > 0, δω > v * q) + 1
            vs[idx, n] = v
        end
    end
end

idx_turn = findlast(isnan, vs[3, :])

for n ∈ 1:idx_turn
    if isnan(vs[2, n])
        vs[2, n] = vs[4, n]
        vs[4, n] = NaN
    end
end

fig = Figure(size=(800, 600), fontsize=24)
ax = Axis(fig[1, 1], xlabel=L"x \ (\mu \text{m})", ylabel=L"v_g \ (\mu \text{m/ ps})")
ylims!(ax, -5, 5)
series!(ax, xs, vs, linewidth=2)
axislegend(ax, position=:rb)
fig
##
x_turn = xs[idx_turn]

vg1 = cubic_spline_interpolation(xs, vs[1, :], extrapolation_bc=Flat())
vg2 = cubic_spline_interpolation(xs, vs[2, :], extrapolation_bc=Flat())
vg3 = cubic_spline_interpolation(xs[idx_turn:end], vcat(0, vs[3, idx_turn+1:end]), extrapolation_bc=Flat())
vg4 = cubic_spline_interpolation(xs[idx_turn:end], vcat(0, vs[4, idx_turn+1:end]), extrapolation_bc=Flat())
##
fig = Figure(size=(800, 600), fontsize=24)
ax = Axis(fig[1, 1], xlabel=L"x \ (\mu \text{m})", ylabel=L"t \ (\text{ps})")

f(x, p, t) = vg4(x)
u0 = x_turn * 0.999

tspan = (0.0, 5)
prob = ODEProblem(f, u0, tspan)
sol4 = solve(prob, Tsit5(), reltol = 1e-8, abstol = 1e-8)

lines!(ax, sol4.u, sol4.t, linewidth=2, color=:red, linestyle=:dash)
fig

f(x, p, t) = vg3(x)
u0 = sol4.u[end]

tspan = (-sol4.t[end], 0)
prob = ODEProblem(f, u0, tspan)
sol3 = solve(prob, Tsit5(), reltol = 1e-8, abstol = 1e-8)
lines!(ax, sol3.u, sol3.t, linewidth=2, color=:red)

f(x, p, t) = vg1(x)
u0 = x_turn * 0.999

tspan = (0, 3)
prob = ODEProblem(f, u0, tspan)
sol1 = solve(prob, Tsit5(), reltol = 1e-8, abstol = 1e-8)
lines!(ax, sol1.u, sol1.t, linewidth=2, color=:green)

tspan = (0, -3)
prob = ODEProblem(f, u0, tspan)
sol1 = solve(prob, Tsit5(), reltol = 1e-8, abstol = 1e-8)
lines!(ax, sol1.u, sol1.t, linewidth=2, color=:green)

f(x, p, t) = vg2(x)
u0 = x_turn * 0.999

tspan = (0, 3)
prob = ODEProblem(f, u0, tspan)
sol2 = solve(prob, Tsit5(), reltol = 1e-8, abstol = 1e-8)
lines!(ax, sol2.u, sol2.t, linewidth=2, color=:blue)

tspan = (0, -3)
prob = ODEProblem(f, u0, tspan)
sol2 = solve(prob, Tsit5(), reltol = 1e-8, abstol = 1e-8)
lines!(ax, sol2.u, sol2.t, linewidth=2, color=:blue)

fig