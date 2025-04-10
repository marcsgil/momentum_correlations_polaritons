using KernelAbstractions

function dispersion(ks, param)
    -im * param.γ / 2 + param.ħ * sum(abs2, ks) / 2param.m - param.δ₀
end

gaussian(x, center, width) = exp(-((x - center) / width)^2)

function potential(xs, param)
    param.V_def * gaussian(xs[1], param.x_def, param.w_def) -
    im * param.V_damp * (gaussian(xs[1], zero(xs[1]), param.w_damp) +
                         gaussian(xs[1], param.L, param.w_damp))
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

noise_func(ψ, param) = √(param.γ / 2 / param.dx)

choose(x1, x2, m) = isone(m) ? x1 : x2

function calculate_momentum_commutators(window1, window2, first_idx1, first_idx2, dx)
    commutators_k = stack(complex(window1) * window2' for a ∈ 1:2, b ∈ 1:2)
    commutators_k[:, :, 1, 1] .= sum(abs2, window1) / dx
    commutators_k[:, :, 2, 2] .= sum(abs2, window2) / dx

    off_diag_comm = zero(commutators_k[:, :, 1, 2])

    for n′ ∈ axes(off_diag_comm, 2)
        n = n′ + first_idx2 - first_idx1
        if n ∈ axes(off_diag_comm, 1)
            off_diag_comm[n, n′] = window1[n] * conj(window2[n′]) / dx
        end
    end

    fft!(off_diag_comm, 1)
    bfft!(off_diag_comm, 2)
    commutators_k[:, :, 1, 2] .= off_diag_comm
    commutators_k[:, :, 2, 1] .= adjoint(off_diag_comm)

    commutators_k
end

function calculate_position_commutators(one_point, dx)
    commutators_x = similar(one_point)
    commutators_x[:, :, 1, 1] .= 1 / dx
    commutators_x[:, :, 2, 2] .= 1 / dx
    commutators_x[:, :, 1, 2] .= one(view(commutators_x, :, :, 1, 2)) ./ dx
    commutators_x[:, :, 2, 1] .= view(commutators_x, :, :, 1, 2)

    commutators_x
end

otherindex(x) = mod(x, 2) + 1

function calculate_g2(first_order, second_order, commutators)
    G2 = second_order .+ (commutators[:, :, 1, 1] .* commutators[:, :, 2, 2] .+ commutators[:, :, 1, 2] .* commutators[:, :, 2, 1]) ./ 4

    for n ∈ axes(first_order, 4), m ∈ axes(first_order, 3)
        G2 .-= first_order[:, :, m, n] .* commutators[:, :, otherindex(m), otherindex(n)] ./ 2
    end

    n1 = first_order[:, :, 1, 1] - commutators[:, :, 1, 1] / 2
    n2 = first_order[:, :, 2, 2] - commutators[:, :, 2, 2] / 2

    real(G2 ./ n1 ./ n2)
end