using Revise, GeneralizedGrossPitaevskii, CairoMakie, LinearAlgebra, CUDA, CUDA.CUFFT, Statistics, ProgressMeter, KernelAbstractions, HDF5
include("polariton_funcs.jl")

function dispersion(ks, param)
    -im * param.γ / 2 + param.ħ * sum(abs2, ks) / 2param.m - param.δ₀
end

function potential(rs, param)
    -param.V_def * exp(-sum(abs2, rs) / param.w_def^2) +
    param.V_damp * damping_potential(rs, -param.L / 2, param.L / 2, param.w_damp)
end

function A(t, Amax, t_cycle, t_freeze)
    _t = ifelse(t > t_freeze, t_freeze, t)
    val = Amax * _t * (t_cycle - _t) * 4 / t_cycle^2
    val < 0 ? zero(val) : val
end

function pump(x, param, t)
    a = A(t, param.Amax, param.t_cycle, param.t_freeze)
    if x[1] ≤ -param.L * 0.9 / 2 || x[1] ≥ -10
        a *= 0
    elseif -param.L * 0.9 / 2 < x[1] ≤ -param.L * 0.85 / 2
        a *= 6
    end
    a * cis(mapreduce(*, +, param.k_pump, x))
end

noise_func(ψ, param) = √(param.γ / 2 / param.δL)

nonlinearity(ψ, param) = param.g * abs2(ψ)

# Space parameters
L = 800.0f0
lengths = (L,)
N = 1024
δL = L / N
rs = range(; start=-L / 2, step=L / N, length=N)

# Polariton parameters
ħ = 0.6582f0 #meV.ps
γ = 0.047f0 / ħ
m = ħ^2 / 2.5f0
g = 0.0003f0 / ħ
δ₀ = 0.49 / ħ

# Potential parameters
V_damp = 100.0f0
w_damp = 5.0f0
V_def = -0.85f0 / ħ
w_def = 0.75f0

# Pump parameters
k_pump = 0.25f0
δ = δ₀ - ħ * k_pump^2 / 2m

# Bistability cycle parameters
Imax = 90.0f0
Amax = √Imax
t_cycle = 300.0f0
t_freeze = 288.0f0

δt = 2.0f-2

# Full parameter tuple
param = (; δ₀, m, γ, ħ, L, g, V_damp, w_damp, V_def, w_def,
    Amax, t_cycle, t_freeze, δL, k_pump)

u0_empty = CUDA.zeros(ComplexF32, N)
prob_steady = GrossPitaevskiiProblem(u0_empty, lengths; dispersion, potential, nonlinearity, pump, param)
tspan_steady = (0, 800.0f0)
solver_steady = StrangSplittingC(512, δt)
ts_steady, sol_steady = solve(prob_steady, solver_steady, tspan_steady);

steady_state = sol_steady[:, end]
heatmap(rs, ts_steady, Array(abs2.(sol_steady)))
##
with_theme(theme_latexfonts()) do
    fig = Figure(; fontsize=20)
    ax = Axis(fig[1, 1], xlabel=L"x", ylabel=L"n")
    offset = 150
    J = N÷2-offset:N÷2+offset
    lines!(ax, rs[J], g * Array(abs2.(steady_state[J])), linewidth=4)
    fig
end
##
ks = GeneralizedGrossPitaevskii.reciprocal_grid(prob_steady)[1]

ϕ₊ = angle.(steady_state[2:end])
ϕ₋ = angle.(steady_state[1:end-1])
∇ϕ = mod2pi.(ϕ₊ - ϕ₋) / δL
v = ħ * ∇ϕ / m

ψ₀ = steady_state[2:end-1]
ψ₊ = steady_state[3:end]
ψ₋ = steady_state[1:end-2]
∇ψ = (ψ₊ + ψ₋ - 2ψ₀) / param.δL^2
δ_vec = δ₀ .+ ħ * real(∇ψ ./ ψ₀) / 2m

c = [speed_of_sound(abs2(ψ), δ, g, ħ, m) for (ψ, δ) ∈ zip(Array(steady_state[2:end-1]), Array(δ_vec))]

with_theme(theme_latexfonts()) do
    fig = Figure(; fontsize=20)
    ax = Axis(fig[1, 1], xlabel=L"x")
    offset = 100
    J = N÷2-offset:N÷2+offset
    lines!(ax, rs[J], c[J], linewidth=4, color=:blue, label=L"c")
    lines!(ax, rs[J], Array(v[J]), linewidth=4, color=:red, label=L"v")
    axislegend()
    fig
end
##
ns_theo = LinRange(0, 1500, 512)
Is_theo = eq_of_state.(ns_theo, δ, g, γ)

n_up = abs2(Array(sol_steady)[N÷4, end])
n_down = abs2(Array(sol_steady)[3N÷4, end])

with_theme(theme_latexfonts()) do
    fig = Figure(fontsize=16)
    ax = Axis(fig[1, 1]; xlabel="I", ylabel="n")
    lines!(ax, Is_theo, ns_theo, color=:blue, linewidth=4, label="Theoretical")
    A_stop = A(Inf, param.Amax, param.t_cycle, param.t_freeze)
    scatter!(ax, abs2(A_stop), abs2(Array(steady_state)[N÷4, end]), color=:black, markersize=16)
    fig
end
##
function one_point_corr!(dest, sol)
    backend = get_backend(dest)

    @kernel function kernel!(dest, sol)
        j = @index(Global)
        x = zero(eltype(dest))
        for k ∈ axes(sol, 2)
            x += abs2(sol[j, k])
        end
        dest[j] += x
    end

    kernel!(backend, 64)(dest, sol, ndrange=length(dest))
    KernelAbstractions.synchronize(backend)
end

function two_point_corr!(dest, sol)
    backend = get_backend(dest)

    @kernel function kernel!(dest, sol)
        j, k = @index(Global, NTuple)
        x = zero(eltype(dest))
        for m ∈ axes(sol, 2)
            x += abs2(sol[j, m]) * abs2(sol[k, m])
        end
        dest[j, k] += x
    end

    kernel!(backend, 64)(dest, sol, ndrange=size(dest))
    KernelAbstractions.synchronize(backend)
end

function calculate_correlation(steady_state, lengths, batchsize, nbatches, tspan, δt; param, kwargs...)
    u0_steady = stack(steady_state for _ ∈ 1:batchsize)
    noise_prototype = similar(u0_steady)

    prob = GrossPitaevskiiProblem(u0_steady, lengths; noise_prototype, param, kwargs...)
    solver = StrangSplittingC(1, δt)

    one_point_r = similar(steady_state, real(eltype(steady_state)))
    two_point_r = similar(steady_state, real(eltype(steady_state)), size(steady_state, 1), size(steady_state, 1))

    one_point_k = similar(steady_state, real(eltype(steady_state)))
    two_point_k = similar(steady_state, real(eltype(steady_state)), size(steady_state, 1), size(steady_state, 1))

    for array in (one_point_r, two_point_r, one_point_k, two_point_k)
        fill!(array, zero(eltype(array)))
    end

    for batch ∈ 1:nbatches
        @info "Batch $batch"
        ts, _sol = solve(prob, solver, tspan; save_start=false)
        sol = dropdims(_sol, dims=3)
        ft_sol = fftshift(fft(ifftshift(sol, 1), 1), 1)

        one_point_corr!(one_point_r, sol)
        two_point_corr!(two_point_r, sol)
        one_point_corr!(one_point_k, ft_sol)
        two_point_corr!(two_point_k, ft_sol)
    end

    for array in (one_point_r, two_point_r, one_point_k, two_point_k)
        array ./= nbatches * batchsize
    end

    #Real space
    δ = one(two_point_r)
    factor = 1 / 2param.δL
    n = one_point_r .- factor

    G2_r = (two_point_r .- factor .* (1 .+ δ) .* (n .+ n' .+ factor)) ./ (n .* n')

    h5open("g2_momentum.h5", "cw") do file
        file["G2_r"] = Array(G2_r)
    end

    #Momentum space
    δ = one(two_point_k)
    factor = 1
    n = one_point_k .- factor

    G2_k = (two_point_k .- factor .* (1 .+ δ) .* (n .+ n' .+ factor)) ./ (n .* n')

    h5open("g2_momentum.h5", "cw") do file
        file["G2_k"] = Array(G2_k)
    end

    G2_r, G2_k
end

tspan_noise = (0.0f0, 50.0f0) .+ tspan_steady[end]

G2_r, G2_k = calculate_correlation(steady_state, lengths, 10^5, 100, tspan_noise, δt;
    dispersion, potential, nonlinearity, pump, param, noise_func)
##
J = N÷2-130:N÷2+130
ks = range(; start=-π / δL, step=2π / (N * δL), length=N)

with_theme(theme_latexfonts()) do
    fig = Figure(; size=(730, 600), fontsize=20)
    ax = Axis(fig[1, 1], aspect=DataAspect(), xlabel=L"k", ylabel=L"k\prime")
    #hm = heatmap!(ax, rs[J], rs[J], (Array(real(G2)[J, J]) .- 1) * 1e5, colorrange=(-5, 5), colormap=:inferno)
    hm = heatmap!(ax, ks[J] .- k_pump, ks[J].- k_pump, (Array(real(G2_k)[J, J]) .- 1) * 1e3, colorrange=(-3, 3), colormap=:inferno)
    Colorbar(fig[1, 2], hm, label=L"g_2(k, k\prime) -1 \ \ ( \times 10^{-3})")
    #lines!(ax, corr_up, corr_down, linewidth=4)
    #save("dev_env/g2m1.pdf", fig)
    fig
end
##