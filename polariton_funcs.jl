function eq_of_state(n, δ, g, γ)
    n * (γ^2 / 4 + (g * n - δ)^2)
end

function dispersion_relation(k, kₚ, g, n₀, δ, m, branch::Bool)
    v = ħ * kₚ / m
    pm = branch ? 1 : -1
    gn₀ = g * n₀
    v * k + pm * √((2gn₀ - δ + ħ * k^2 / 2m)^2 - (gn₀)^2)
end

function speed_of_sound(n, δ, g, ħ, m)
    2g * n ≥ δ ? √(ħ * (2g * n - δ) / m) : NaN
end