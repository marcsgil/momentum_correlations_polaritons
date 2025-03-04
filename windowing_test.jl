using GeneralizedGrossPitaevskii, CairoMakie, FFTW, Statistics, LinearAlgebra, CUDA
include("polariton_funcs.jl")
include("io.jl")
include("TruncatedWigner/equations.jl")

# Space parameters
L = 20.0f0
lengths = (L,)
N = 64
δL = L / N
rs = range(; start=-L / 2, step=L / N, length=N)

# Polariton parameters
ħ = 0.6582f0 #meV.ps
γ = 0.1f0 / ħ
m = ħ^2 / 2.5f0
δ₀ = 0.49 / ħ

δt = 4.0f0

# Full parameter tuple
param = (; δ₀, m, γ, ħ, L, δL, N, δt)

u0 = CUDA.zeros(ComplexF32, N, 10^6)
noise_prototype = similar(u0)
prob = GrossPitaevskiiProblem(u0, lengths; dispersion, param, noise_func, noise_prototype)
tspan = (0, 200.0f0)
solver = StrangSplittingC(1, δt)
ts, _sol = GeneralizedGrossPitaevskii.solve(prob, solver, tspan, save_start=false);

sol = dropdims(_sol, dims=3)
##


GeneralizedGrossPitaevskii.direct_grid(prob)
##
function correlation(sol, rs, window1, par1, window2, par2)
    sol1 = sol .* map(x -> window1(x, par1), rs)
    sol2 = sol .* map(x -> window2(x, par2), rs)

    ft_sol1 = ifftshift(fft(fftshift(sol1, 1), 1), 1)
    ft_sol2 = ifftshift(fft(fftshift(sol2, 1), 1), 1)

    g1 = similar(sol, size(sol, 1), size(sol, 1))

    Threads.@threads for j in axes(g1, 2)
        j_slice = view(ft_sol2, j, :)
        for i in axes(g1, 1)
            i_slice = view(ft_sol1, i, :)
            g1[i, j] = j_slice ⋅ i_slice / length(sol)
        end
    end

    g1
end

function analytic_commutation(L, N, window1, par1, window2, par2)
    rs = range(; start=-L / 2, step=L / N, length=N)
    ks = range(-π / L, step=2π / L, length=N)
    [sum(rs) do r
        cis((k - k′) * r) * window1(r, par1) * conj(window2(r, par2))
    end for k in ks, k′ in ks] / (rs[end] - rs[1]) / 2
end

window(x, (x0, w)) = exp(-(x - x0)^2 / w^2)
#window(x, par) = one(x)

x0 = 8
args = (window, (x0, 5.0), window, (-x0, 5.0))

corr = correlation(Array(sol), rs, args...)
an_corr = analytic_commutation(L, N, args...)

isapprox(corr, an_corr; rtol=1e-1)

M, J = findmax(abs, corr - an_corr)

corr[J]
an_corr[J]

with_theme(theme_latexfonts()) do
    fig = Figure(;size=(800,400))
    colorrange = extrema(abs, corr)
    for (n, img) ∈ enumerate((corr, an_corr))
        ax = Axis(fig[1, n], aspect=DataAspect())
        heatmap!(ax, abs.(img), colorrange=colorrange)
    end
    fig
end
##
using GeneralizedGrossPitaevskii

