using KernelAbstractions, FFTW

function dispersion(ks, param)
    param.ħ * sum(abs2, ks) / 2param.m - param.δ₀
end

gaussian(x, center, width) = exp(-((x - center) / width)^2)

function loss(xs, param)
    param.γ / 2 + param.V_damp * (gaussian(xs[1], zero(xs[1]), param.w_damp) +
                                  gaussian(xs[1], param.L, param.w_damp))
end

function potential(xs, param)
    param.V_def * gaussian(xs[1], param.x_def, param.w_def) - im * loss(xs, param)
end

function half_pump(x, Fmax, Fmin, k, w, x0)
    ((Fmax - Fmin) * sech((x - x0) / w) + Fmin) * cis(k * x)
end

time_dependence(t, param) = (param.extra_intensity * exp(-t / param.decay_time) + 1)

function pump(x, param, t)
    if x[1] < param.divide
        k = param.k_up
        F = param.F_up
        x0 = 0
    else
        k = param.k_down
        F = param.F_down
        x0 = param.L
    end

    ((param.F_max - F) * sech((x[1] - x0) / param.w_pump) + F) * cis(k * x[1]) * time_dependence(t, param)
end

nonlinearity(ψ, param) = param.g * (abs2(first(ψ)) - 1 / param.dx)

position_noise_func(ψ, xs, param) = √(loss(xs, param) / param.dx)

function calculate_momentum_commutators(window1, window2, first_idx1, first_idx2, dx)
    c11 = sum(abs2, window1) / dx
    c22 = sum(abs2, window2) / dx

    c12 = zero(complex(window1) * window2')

    for n′ ∈ axes(c12, 2)
        n = n′ + first_idx2 - first_idx1
        if n ∈ axes(c12, 1)
            c12[n, n′] = window1[n] * conj(window2[n′]) / dx
        end
    end

    fft!(c12, 1)
    bfft!(c12, 2)

    c11, c22, c12
end

function calculate_position_commutators(N, dx)
    c11 = 1 / dx
    c22 = 1 / dx
    c11, c22, Array(I(N)) ./ dx
end

otherindex(x) = mod(x, 2) + 1

function calculate_g2(averages, commutators)
    n1, n2, G1, G2 = averages
    c11, c22, c12 = commutators

    numerator = (
        G2 + (c11 * c22 .+ abs2.(c12)) / 4
        -
        (c11 .* n1 .+ c22 .* n2') / 2
        -
        real(c12 .* G1)
    )

    denominator = (n1 .- c11 / 2) * (n2 .- c22 / 2)'
    numerator ./ denominator
end

function calculate_g2m1(averages, commutators)
    μ1, μ2, G1, σ² = averages
    c11, c22, c12 = commutators

    n1 = μ1 .- c11 / 2
    n2 = μ2 .- c22 / 2

    (σ² - real(conj(c12) .* G1) + abs2.(c12) / 4) ./ (n1 * n2')
end