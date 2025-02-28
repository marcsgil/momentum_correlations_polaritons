using CairoMakie, HDF5, FFTW
include("tracing.jl")
include("io.jl")
include("polariton_funcs.jl")

G2_r, G2_k, param, steady_state = h5open("/home/stagios/Marcos/LEON_Marcos/Users/Marcos/MomentumCorrelations/Old/correlation_old.h5", "r") do file
    group = file["no_support"]
    read(group, "G2_r"), read(group, "G2_k"), read_parameters(group), read(group, "steady_state")
end

N = param.N
L = param.L
δL = param.δL
rs = range(; start=-param.L / 2, step=param.δL, length=param.N)

m = param.m
δ₀ = param.δ₀
g = param.g
ħ = param.ħ
k_up = param.k_up
k_down = √(2m * δ₀ / ħ)

n_up = abs2(Array(steady_state)[N÷4])

power = 5
with_theme(theme_latexfonts()) do
    fig = Figure(; size=(730, 600), fontsize=20)
    ax = Axis(fig[1, 1], aspect=DataAspect(), xlabel=L"x", ylabel=L"x\prime")
    xlims!(ax, (-150, 150))
    ylims!(ax, (-150, 150))
    hm = heatmap!(ax, rs, rs, (Array(real(G2_r)) .- 1) * 10^power, colorrange=(-5, 5), colormap=:inferno)
    Colorbar(fig[1, 2], hm, label=L"g_2(x, x\prime) -1 \ \ ( \times 10^{-%$power})")
    #save("Plots/position_correlations.pdf", fig)
    fig
end
##
ks1 = LinRange(-1, 1, 512)
ks2 = LinRange(-1.5, 1.5, 512)

param_up = (n_up, g, δ₀, k_up, ħ, m)
ω₊_up = [dispersion_relation(k, param_up..., true) for k ∈ ks1]
ω₋_up = [dispersion_relation(k, param_up..., false) for k ∈ ks1]

param_down = (nextfloat(0f0), g, δ₀, k_down, ħ, m)
ω₊_down = [dispersion_relation(k, param_down..., true) for k ∈ ks2]
ω₋_down = [dispersion_relation(k, param_down..., false) for k ∈ ks2]

Ω = 0.6
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


with_theme(theme_latexfonts()) do
    fig = Figure(fontsize=20, size=(800, 400))
    ax1 = Axis(fig[1, 1]; xlabel=L"\delta k", ylabel=L"\delta \omega")
    ax2 = Axis(fig[1, 2]; xlabel=L"\delta k")

    hideydecorations!(ax2, grid=false)
    ylims!(ax1, (-1, 1))
    ylims!(ax2, (-1, 1))

    lines!(ax1, ks1, ω₊_up, linewidth=4)
    lines!(ax1, ks1, ω₋_up, linewidth=4)
    lines!(ax2, ks2, ω₊_down, linewidth=4)
    lines!(ax2, ks2, ω₋_down, linewidth=4)

    for ax ∈ (ax1, ax2)
        hlines!(ax, [Ω, -Ω], linestyle=:dash, color=:black)
    end

    scatter!(ax1, k_up_out, Ω, color=:black, markersize=16)
    text!(ax1, k_up_out, Ω, text=L"u1_\text{out}", fontsize=24, align=(:right, :bottom), offset=(-10, 0))
    scatter!(ax1, k_up_in, Ω, color=:black, markersize=16)
    text!(ax1, k_up_in, Ω, text=L"u1_\text{in}", fontsize=24, align=(:right, :bottom), offset=(-10, 0))
    scatter!(ax1, k_star_up_out, -Ω, color=:black, markersize=16)
    text!(ax1, k_star_up_out, -Ω, text=L"u1_\text{out}^*", fontsize=24, align=(:right, :bottom), offset=(-10, 0))
    scatter!(ax1, k_star_up_in, -Ω, color=:black, markersize=16)
    text!(ax1, k_star_up_in, -Ω, text=L"u1_\text{in}^*", fontsize=24, align=(:right, :bottom), offset=(-10, 0))

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

    #save("Plots/dispersion_relation.pdf", fig)
    fig
end
##
#Hawking (u1 out, d2 out)
param1 = (param_up..., true)
param2 = (param_down..., false)

k_ext1, ω_ext1 = find_extrema(dispersion_relation, (-1, 1), param1...)
k_ext2, ω_ext2 = find_extrema(dispersion_relation, (0.1, 1), param2...)

k1_min = find_zero(k -> dispersion_relation(k, param1...) - ω_ext2, (-1, k_ext1))
k2_min = find_zero(k -> dispersion_relation(k, param2...) - ω_ext1, (0, k_ext2))

bracket1 = (k1_min * 0.9999, k_ext1)
bracket2 = (k2_min, k_ext2)

corr_up_u1d2, corr_down_u1d2 = correlate(param1, bracket1, param2, bracket2, 128, false)
## (u out, d1 out)
param1 = (param_up..., true)
param2 = (param_down..., true)

k_ext1, ω_ext1 = find_extrema(dispersion_relation, (-1, 1), param1...)

k1_min = find_zero(k -> dispersion_relation(k, param1...) - ω_ext2, (-1, k_ext1))

bracket1 = (-0.8, k_ext1)
bracket2 = (0.01, 1)

corr_up_u1d1, corr_down_u1d1 = correlate(param1, bracket1, param2, bracket2, 128, false)
## (d1, d2)
param1 = (param_down..., true)
param2 = (param_down..., false)

k_ext1, ω_ext1 = find_extrema(dispersion_relation, (-1, -0.1), param1...)
k_ext2, ω_ext2 = find_extrema(dispersion_relation, (0.1, 1), param2...)

k1_max = find_zero(k -> dispersion_relation(k, param1...) - ω_ext2, (0.1, 1))
k2_min = find_zero(k -> dispersion_relation(k, param2...) - ω_ext1, (-1, -0.1))

bracket1 = (0, k1_max)
bracket2 = (0, k_ext2)

corr_d1d2, corr_d1d2′ = correlate(param1, bracket1, param2, bracket2, 128, false)
## (d1_star, d2_star)
param1 = (param_down..., true)
param2 = (param_down..., false)

k_ext1, ω_ext1 = find_extrema(dispersion_relation, (-1, -0.1), param1...)
k_ext2, ω_ext2 = find_extrema(dispersion_relation, (0.1, 1), param2...)

k1_max = find_zero(k -> dispersion_relation(k, param1...) - ω_ext2, (0.1, 1))
k2_min = find_zero(k -> dispersion_relation(k, param2...) - ω_ext1, (-1, -0.1))

bracket1 = (k_ext1, 0)
bracket2 = (k2_min, 0)

corr_d1_star_d2_star, corr_d1_star_d2_star′ = correlate(param1, bracket1, param2, bracket2, 128, false)
## (d2, d2_star)
param1 = (param_down..., true)
param2 = (param_down..., false)

k_ext1, ω_ext1 = find_extrema(dispersion_relation, (-1, -0.1), param1...)
k_ext2, ω_ext2 = find_extrema(dispersion_relation, (0.1, 1), param2...)

k1_max = find_zero(k -> dispersion_relation(k, param1...) - ω_ext2, (0.1, 1))
k2_min = find_zero(k -> dispersion_relation(k, param2...) - ω_ext1, (-1, -0.1))

dispersion_relation(k1_max, param1...)

bracket1 = (0, k1_max)
bracket2 = (k2_min, 0)

corr_d2d2_star, corr_d2d2_star′ = correlate(param1, bracket1, param2, bracket2, 128, true)
##
ks = range(; start=-π / param.δL, step=2π / (size(G2_k, 1) * param.δL), length=size(G2_k, 1))
power = 3

ticks = [0.0]
ticklabels = [L"%$tick" for tick in ticks]
for (k, label) in zip((k_up, k_down, -k_down), (L"k_{\text{up}}", L"k_{\text{down}}", L"-k_{\text{down}}"))
    push!(ticks, k)
    push!(ticklabels, label)
end


with_theme(theme_latexfonts()) do
    fig = Figure(; size=(900, 600), fontsize=20)
    ax = Axis(fig[1, 1]; aspect=DataAspect(), xlabel=L"k", ylabel=L"k\prime", xticks=(ticks, ticklabels), yticks=(ticks, ticklabels))
    xlims!(ax, (-0.6, 1.3))
    ylims!(ax, (-0.6, 1.3))
    hm = heatmap!(ax, ks, ks, (G2_k .- 1) * 10^3, colorrange=(-1, 1), colormap=:inferno)
    Colorbar(fig[1, 2], hm, label=L"g_2(k, k\prime) -1 \ \ ( \times 10^{-%$power})")
    for line_func! in (hlines!, vlines!)
        for k in (k_up, k_down, -k_down)
            line_func!(ax, k, color=:green, linestyle=:dash)
        end
    end
    lines!(ax, corr_down_u1d2 .+ k_down, corr_up_u1d2 .+ k_up, linewidth=4, color=:blue, linestyle=:dash, label=L"u1_{\text{out}} \leftrightarrow d2_{\text{out}}")
    lines!(ax, corr_down_u1d1 .+ k_down, corr_up_u1d1 .+ k_up, linewidth=4, color=:red, linestyle=:dash, label=L"u1_{\text{out}} \leftrightarrow d1_{\text{out}}")
    #lines!(ax, -corr_down_u1d2 .+ k_down, -corr_up_u1d2 .+ k_up, linewidth=4, color=:blue, linestyle=:dash, label=L"u1_{\text{out}} \leftrightarrow d2_{\text{out}}")
    #lines!(ax, -corr_down_u1d1 .+ k_down, -corr_up_u1d1 .+ k_up, linewidth=4, color=:red, linestyle=:dash, label=L"u1_{\text{out}} \leftrightarrow d1_{\text{out}}")
    lines!(ax, corr_d1d2 .+ k_down, corr_d1d2′ .+ k_down, linewidth=4, color=:green, linestyle=:dash, label=L"d1_{\text{out}} \leftrightarrow d2_{\text{out}}")
    lines!(ax, corr_d1_star_d2_star′ .+ k_down, corr_d1_star_d2_star .+ k_down, linewidth=4, color=:purple, linestyle=:dash, label=L"d1_{\text{out}}^* \leftrightarrow d2_{\text{out}}^*")
    lines!(ax, corr_d2d2_star .+ k_down, corr_d2d2_star′ .+ k_down, linewidth=4, color=:orange, linestyle=:dash, label=L"d1_{\text{out}} \leftrightarrow d1_{\text{out}}^*")
    Legend(fig[1, 3], ax)

    #save("Plots/momentum_correlations.pdf", fig)
    fig
end
##
with_theme(theme_latexfonts()) do
    fig = Figure(; size=(850, 600), fontsize=20)
    ax = Axis(fig[1, 1], aspect=DataAspect(), xlabel=L"k", ylabel=L"k\prime")
    hm = heatmap!(ax, (G2_k .- 1) * 10^3, colorrange=(-1, 1), colormap=:inferno)
    Colorbar(fig[1, 2], hm, label=L"g_2(k, k\prime) -1 \ \ ( \times 10^{-%$power})")
    fig
end
##
dft(x) = fftshift(fft(ifftshift(x)))

len = 30
J = (-len:len) .+ 470
K = (-len:len) .+ 780
power = 4

x_period = range(start=-len * param.δL, step=param.δL, length=2 * len)

with_theme(theme_latexfonts()) do
    fig = Figure(; size=(1400, 600), fontsize=20)
    ax = Axis(fig[1, 1], aspect=DataAspect(), xlabel=L"k", ylabel=L"k\prime")
    hm = heatmap!(ax, ks[J], ks[K], (G2_k[J, K] .- 1) * 10^power, colorrange=(-2, 2), colormap=:inferno)
    Colorbar(fig[1, 2], hm, label=L"g_2(k, k\prime) -1 \ \ ( \times 10^{-%$power})")
    ax2 = Axis(fig[1, 3], aspect=DataAspect(), xlabel=L"x", ylabel=L"x\prime")
    heatmap!(ax2, x_period, x_period, (abs2.(dft(G2_k[J, K] .- 1))), colormap=:inferno)

    #save("period_close_to_hawking.png", fig)
    fig
end
##
len = 30
J = (-len:len) .+ 390
K = (-len:len) .+ 460
power = 4

x_period = range(start=-len * param.δL, step=param.δL, length=2 * len)

with_theme(theme_latexfonts()) do
    fig = Figure(; size=(1400, 600), fontsize=20)
    ax = Axis(fig[1, 1], aspect=DataAspect(), xlabel=L"k", ylabel=L"k\prime")
    hm = heatmap!(ax, ks[J], ks[K], (G2_k[J, K] .- 1) * 10^power, colorrange=(-2, 2), colormap=:inferno)
    Colorbar(fig[1, 2], hm, label=L"g_2(k, k\prime) -1 \ \ ( \times 10^{-%$power})")
    ax2 = Axis(fig[1, 3], aspect=DataAspect(), xlabel=L"x", ylabel=L"x\prime")
    heatmap!(ax2, x_period, x_period, (abs2.(dft(G2_k[J, K] .- 1) * 10^4)), colormap=:inferno)

    #save("period_close_diagonal.png", fig)
    fig
end