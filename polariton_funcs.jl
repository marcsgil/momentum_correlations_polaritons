using LinearAlgebra, Roots

detuning(δ₀, K, ħ, m) = δ₀ - ħ * K^2 / 2m

function eq_of_state(n, g, δ₀, K, ħ, m, γ)
    δ = detuning(δ₀, K, ħ, m)
    n * (γ^2 / 4 + (g * n - δ)^2)
end

function dispersion_relation(k, n, g, δ₀, K, ħ, m, branch::Bool)
    gn = g * n
    v = ħ * K / m
    δ = detuning(δ₀, K, ħ, m)
    v * k + (2branch - 1) * √((2gn - δ + ħ * k^2 / 2m)^2 - (gn)^2)
end

function speed_of_sound(n, g, δ₀, K, ħ, m)
    δ = detuning(δ₀, K, ħ, m)
    gn = g * n
    #2gn ≥ δ ? √(ħ * (2gn - δ) / m) : NaN
    √(ħ * gn / m)
end

function velocity(steady_state, ħ, m, δL)
    ħ * mod2pi.(finite_difference_grad(angle.(steady_state))) / m / δL
end

function finite_difference_grad(N::Integer)
    Tridiagonal(-ones(N - 1), ones(N), zeros(N - 1))
end

function finite_difference_grad(ψ)
    finite_difference_grad(length(ψ)) * ψ
end

function finite_difference_lap(N::Integer)
    Tridiagonal(ones(N - 1), -2ones(N), ones(N - 1))
end

function finite_difference_lap(ψ)
    finite_difference_lap(length(ψ)) * ψ
end