using GeneralizedGrossPitaevskii, CUDA, CUDA.CUFFT, KernelAbstractions, CairoMakie
include("polariton_funcs.jl")
include("io.jl")

function dispersion(ks, param)
    -im * param.γ / 2 + param.ħ * sum(abs2, ks) / 2param.m - param.δ₀
end

function potential(rs, param)
    param.V_def * exp(-sum(abs2, rs) / param.w_def^2) +
    param.V_damp * damping_potential(rs, -param.L / 2, param.L / 2, param.w_damp)
end

nonlinearity(ψ, param) = param.g * abs2(ψ)

function A(t, Amax, t_cycle, t_freeze)
    _t = ifelse(t > t_freeze, t_freeze, t)
    val = Amax * _t * (t_cycle - _t) * 4 / t_cycle^2
    val < 0 ? zero(val) : val
end

function pump(x, param, t)
    a = A(t, param.Amax, param.t_cycle, param.t_freeze)

    if abs(x[1]) ≥ param.L * 0.85 / 2
        a *= 0
    elseif -param.L * 0.80 / 2 ≥ x[1] > -param.L * 0.85 / 2
        a *= 6
    end

    if x[1] > param.divide
        a *= param.factor
    end

    k = x[1] < param.divide ? param.k_up : param.k_down

    a * cis(mapreduce(*, +, k, x))
end

noise_func(ψ, param) = √(param.γ / 2 / param.δL)

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

function get_correlation_buffers(steady_state)
    one_point = real(zero(steady_state))
    two_point = one_point * one_point'
    one_point, two_point
end

function calculate_correlation(steady_state, lengths, batchsize, nbatches, tspan, δt, saving_path, group_name; param, show_progress=true, kwargs...)
    file = h5open(saving_path, "cw")
    group = create_group(file, group_name)
    write_parameters!(group, param)
    group["steady_state"] = Array(steady_state)

    u0 = stack(steady_state for _ ∈ 1:batchsize)
    noise_prototype = similar(u0)

    prob = GrossPitaevskiiProblem(u0, lengths; noise_prototype, param, kwargs...)
    solver = StrangSplittingC(1, δt)

    one_point_r, two_point_r = get_correlation_buffers(steady_state)
    one_point_k, two_point_k = get_correlation_buffers(steady_state)

    for batch ∈ 1:nbatches
        @info "Batch $batch"
        ts, _sol = GeneralizedGrossPitaevskii.solve(prob, solver, tspan; save_start=false, show_progress)
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

    for (one_point, two_point, factor, type) ∈ ((one_point_r, two_point_r, 1 / 2param.δL, "r"), (one_point_k, two_point_k, 1, "k"))
        δ = one(two_point)
        n = one_point .- factor
        G2 = (two_point .- factor .* (1 .+ δ) .* (n .+ n' .+ factor)) ./ (n .* n')
        h5open(saving_path, "cw") do file
            group["G2_"*type] = Array(G2)
        end
    end
    close(file)
end

# Space parameters
L = 1600.0f0
lengths = (L,)
N = 1024
δL = L / N
rs = range(; start=-L / 2, step=L / N, length=N)

# Polariton parameters
ħ = 0.6582f0 #meV.ps
γ = 0.047f0 / ħ
m = ħ^2 / 2.5f0
#m = 1 / 18f0
g = 0.0003f0 / ħ
δ₀ = 0.49 / ħ

# Potential parameters
V_damp = 100.0f0
w_damp = 10.0f0
V_def = 0.85f0 / ħ
w_def = 0.75f0

# Pump parameters
k_up = 0.25f0
k_down = 0.55f0
divide = -7
factor = 0

# Bistability cycle parameters
Imax = 90.0f0
Amax = √Imax
t_cycle = 300.0f0
t_freeze = 288.0f0

δt = 2.0f-2

# Full parameter tuple
param = (; δ₀, m, γ, ħ, L, g, V_damp, w_damp, V_def, w_def,
    Amax, t_cycle, t_freeze, δL, N, k_down, k_up, divide, factor)

u0 = CUDA.zeros(ComplexF32, N)
prob = GrossPitaevskiiProblem(u0, lengths; dispersion, potential, nonlinearity, pump, param)
tspan = (0, 1200.0f0)
solver = StrangSplittingC(512, δt)
ts, sol = GeneralizedGrossPitaevskii.solve(prob, solver, tspan);
heatmap(rs, ts, Array(abs2.(sol)))
##
steady_state = sol[:, end]
n = Array(abs2.(steady_state))
n_up = n[N÷4]
n_down = n[3N÷4]

with_theme(theme_latexfonts()) do
    fig = Figure(; fontsize=20)
    ax = Axis(fig[1, 1], xlabel=L"x", ylabel=L"gn")
    offset = 300
    J = N÷2-offset:N÷2+offset
    lines!(ax, rs[J], g * n[J], linewidth=4)
    fig
end
##
v = velocity(Array(steady_state), ħ, m, δL)
c = map((n, v) -> speed_of_sound(n, g, δ₀, m * v / ħ, ħ, m), n, v)

with_theme(theme_latexfonts()) do
    fig = Figure(; fontsize=20)
    ax = Axis(fig[1, 1], xlabel=L"x")
    offset = 100
    J = N÷2-offset:N÷2+offset
    lines!(ax, rs[J], c[J], linewidth=4, color=:blue, label=L"c")
    lines!(ax, rs[J], v[J], linewidth=4, color=:red, label=L"v")
    axislegend(; position=:lt)
    fig
end
##
ns_up_theo = LinRange(0, 1600, 512)
Is_up_theo = eq_of_state.(ns_up_theo, g, δ₀, k_up, ħ, m, γ)

ns_down_theo = LinRange(0, 800, 512)
Is_down_theo = eq_of_state.(ns_down_theo, g, δ₀, k_down, ħ, m, γ)

with_theme(theme_latexfonts()) do
    fig = Figure(fontsize=16)
    ax = Axis(fig[1, 1]; xlabel="I", ylabel="n")
    lines!(ax, Is_up_theo, ns_up_theo, color=:blue, linewidth=4, label="Upstream")
    lines!(ax, Is_down_theo, ns_down_theo, color=:red, linewidth=4, label="Downstream")
    A_stop = A(Inf, param.Amax, param.t_cycle, param.t_freeze)
    scatter!(ax, abs2(A_stop), n_up, color=:black, markersize=16)
    scatter!(ax, abs2(A_stop * factor), n_down, color=:black, markersize=16)
    axislegend()
    fig
end
##
tspan_noise = (0.0f0, 50.0f0) .+ tspan[end]
saving_path = "correlation.h5"
group_name = "no_support_cut"

calculate_correlation(steady_state, lengths, 10^5, 1, tspan_noise, δt, saving_path, group_name;
    dispersion, potential, nonlinearity, pump, param, noise_func, show_progress=false)