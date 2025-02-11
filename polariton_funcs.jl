using LinearAlgebra

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

function velocity(steady_state, ħ, m, δL)
    ϕ₊ = @view steady_state[2:end]
    ϕ₋ = @view steady_state[1:end-1]
    @. ħ * imag(log(ϕ₊ / ϕ₋)) / m / δL
end

function finite_difference_grad(N::Integer)
    Tridiagonal(ones(N-1), -ones(N), zeros(N-1))
end

function finite_difference_grad(ψ)
    finite_difference_grad(length(ψ)) * ψ
end

function finite_difference_lap(N::Integer)
    Tridiagonal(ones(N-1), -2ones(N), ones(N-1))
end

function finite_difference_lap(ψ)
    finite_difference_lap(length(ψ)) * ψ
end