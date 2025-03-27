using GeneralizedGrossPitaevskii, CairoMakie, CUDA
include("../polariton_funcs.jl")
include("../io.jl")
include("equations.jl")

# Space parameters
L = 1600.0
lengths = (L,)
N = 1024
δL = L / N
rs = range(; start=-L / 2, step=δL, length=N)

# Polariton parameters
ħ = 0.6582 #meV.ps
γ = 0.047 / ħ
m = 1 / 6 # meV.ps^2/μm^2; This is 3×10^-5 the electron mass
g = 3f-4 / ħ
δ₀ = 0.49 / ħ

# Potential parameters
V_damp = 1000.0
w_damp = 30.0
V_def = 0.85 / ħ
w_def = 0.75

# Pump parameters
k_up = 0.25
k_down = 0.55
divide = -7
factor = 0

# Bistability cycle parameters
Amax = 12
t_cycle = 300.0
t_freeze = 291.0

dt = 2.0f-1
nsaves = 512

# Full parameter tuple
param = (; δ₀, m, γ, ħ, L, g, V_damp, w_damp, V_def, w_def,
    Amax, t_cycle, t_freeze, δL, N, k_down, k_up, divide, factor, dt)

u0 = (CUDA.zeros(ComplexF64, N),)
prob = GrossPitaevskiiProblem(u0, lengths; dispersion, potential, nonlinearity, pump, param)
tspan = (0, 1200.0)
alg = StrangSplittingC()
ts, sol = GeneralizedGrossPitaevskii.solve(prob, alg, tspan; dt, nsaves);
steady_state = map(x -> x[:, end], sol)
heatmap(rs, ts, Array(abs2.(sol[1])))
##
n = Array(abs2.(steady_state[1]))
n_up = n[N÷4]
n_down = n[3N÷4]

with_theme(theme_latexfonts()) do
    fig = Figure(; fontsize=20)
    ax = Axis(fig[1, 1], xlabel=L"x", ylabel=L"gn", xticks=(-800:200:800))
    #xlims!(ax, -200, 200)
    #ylims!(ax, -0.01, 0.75)
    lines!(ax, rs, g * n, linewidth=4)
    #save("/home/stagios/Marcos/LEON_Marcos/Users/Marcos/MomentumCorrelations/Plots/TruncatedWigner/densities.pdf", fig)
    fig
end
##
v = velocity(Array(steady_state[1]), ħ, m, δL)
c = map((n, v) -> speed_of_sound(n, g, δ₀, m * v / ħ, ħ, m), n, v)

with_theme(theme_latexfonts()) do
    fig = Figure(; fontsize=20)
    ax = Axis(fig[1, 1], xlabel=L"x")
    xlims!(ax, -200, 200)
    ylims!(ax, 0, 2.5)
    lines!(ax, rs, c, linewidth=4, color=:blue, label=L"c")
    lines!(ax, rs, v, linewidth=4, color=:red, label=L"v")
    axislegend(; position=:lt)
    #save("/home/stagios/Marcos/LEON_Marcos/Users/Marcos/MomentumCorrelations/Plots/TruncatedWigner/velocities.pdf", fig)
    fig
end
##
ns_up_theo = LinRange(0, 1800, 512)
Is_up_theo = eq_of_state.(ns_up_theo, g, δ₀, k_up, ħ, m, γ)

γ^2 * detuning(δ₀, k_up, ħ, m) / 4g

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
    #save("/home/stagios/Marcos/LEON_Marcos/Users/Marcos/MomentumCorrelations/Plots/TruncatedWigner/bistability.pdf", fig)
    fig
end
##
function get_correlation_buffers(proto1, proto2)
    two_point = zero(proto1) * zero(proto2)'
    one_point = stack(two_point for a ∈ 1:2, b ∈ 1:2)
    one_point, two_point
end

function get_support(m, M, rs)
    argmin(n -> abs(rs[n] - m), eachindex(rs)):argmin(n -> abs(rs[n] - M), eachindex(rs))
end

function create_save_group(_steady_state, saving_path, group_name, window_func, support1, support2)
    steady_state = Array.(_steady_state)

    T = real(eltype(first(steady_state)))

    window1 = [T(window_func(n, length(support1))) for n ∈ 0:length(support1)-1]
    window2 = [T(window_func(n, length(support2))) for n ∈ 0:length(support2)-1]

    one_point_r, two_point_r = get_correlation_buffers(first(steady_state), first(steady_state))
    one_point_k, two_point_k = get_correlation_buffers(complex(window1), complex(window2))

    n_ave = 0

    h5open(saving_path, "cw") do file
        group = create_group(file, group_name)
        write_parameters!(group, param)
        group["steady_state"] = stack(steady_state...)
        group["t_steady_state"] = tspan[end]
        group["one_point_r"] = one_point_r
        group["two_point_r"] = two_point_r
        group["one_point_k"] = one_point_k
        group["two_point_k"] = two_point_k
        group["n_ave"] = [n_ave]
        group["window1"] = window1
        group["window2"] = window2
        group["support1"] = Vector(support1)
        group["support2"] = Vector(support2)
    end
end

saving_path = "/home/stagios/Marcos/LEON_Marcos/Users/Marcos/MomentumCorrelations/TruncatedWigner/correlations.h5"
group_name = "hamming_corrected_k_cut"

hamming(n, N) = 0.54 - 0.46 * cospi(2 * n / (N - 1))

support1 = get_support(-10, 190, rs)
support2 = get_support(-190, 10, rs)


#= h5open(saving_path, "cw") do file
    delete_object(file, group_name)
end =#

create_save_group(steady_state, saving_path, group_name, hamming, support1, support2)