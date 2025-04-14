using LinearAlgebra, DSP

detuning(δ₀, K, ħ, m) = δ₀ - ħ * K^2 / 2m

function eq_of_state(n, g, δ₀, K, ħ, m, γ)
    δ = detuning(δ₀, K, ħ, m)
    n * (γ^2 / 4 + (g * n - δ)^2)
end

function dispersion_relation(k, n, g, δ₀, K, ħ, m, branch::Bool)
    gn = g * n
    v = ħ * K / m
    δ = detuning(δ₀, K, ħ, m)
    v * k + (2branch - 1) * real(√complex((2gn - δ + ħ * k^2 / 2m)^2 - (gn)^2))
end

function speed_of_sound(n, g, δ₀, K, ħ, m)
    δ = detuning(δ₀, K, ħ, m)
    gn = g * n
    2gn ≥ δ ? √(ħ * (2gn - δ) / m) : NaN
end

function velocity(steady_state, ħ, m, dx)
    ħ * diff(unwrap(angle.(steady_state))) / m / dx
end