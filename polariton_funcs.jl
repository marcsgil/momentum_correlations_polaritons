using LinearAlgebra, DSP

detuning(δ₀, K, ħ, m) = δ₀ - ħ * K^2 / 2m

function eq_of_state(n, g, δ₀, K, ħ, m, γ)
    δ = detuning(δ₀, K, ħ, m)
    n * (γ^2 / 4 + (g * n - δ)^2)
end

function mass_term(n, g, δ₀, K, ħ, m)
    δ = detuning(δ₀, K, ħ, m)
    μ² = (3g * n - δ) * (g * n - δ)
    if μ² < 0
        return oftype(μ², NaN)
    else
        return μ²
    end
end

function speed_of_sound(n, g, δ₀, K, ħ, m)
    δ = detuning(δ₀, K, ħ, m)
    c² = ħ * (2g * n - δ) / m
    if c² < 0
        return oftype(c², NaN)
    else
        return √c²
    end
end

function squared_fluid_dr(q, μ², c, ħ, m)
    μ² + (c * q)^2 + (ħ * q^2 / 2m)^2
end

function dispersion_relation(q, μ², c, K, ħ, m, branch::Bool)
    v = ħ * K / m
    v * q + (2branch - 1) * √squared_fluid_dr(q, μ², c, ħ, m)
end

function dispersion_relation(q, n, g, δ₀, K, ħ, m, branch::Bool)
    μ² = mass_term(n, g, δ₀, K, ħ, m)
    c = speed_of_sound(n, g, δ₀, K, ħ, m)
    dispersion_relation(q, μ², c, K, ħ, m, branch::Bool)
end

function group_velocity(q, δω, K, c, ħ, m)
    v = ħ * K / m
    v + q * (c^2 + (ħ * q / m)^2 / 2) / (δω - v * q)
end

function first_derivative_operator(N, dx)
    dl = fill(-1 / 2dx, N - 1)
    du = fill(1 / 2dx, N - 1)
    d = zeros(N)
    d[1] = -1 / dx
    d[N] = 1 / dx
    dl[N-1] = - 1/dx
    du[1] = 1 / dx
    Tridiagonal(dl, d, du)
end

function wavenumber(steady_state, dx)
    first_derivative_operator(length(steady_state), dx) * unwrap(angle.(steady_state))
end

function velocity(steady_state, ħ, m, dx)
    ħ * wavenumber(steady_state, dx) / m
end