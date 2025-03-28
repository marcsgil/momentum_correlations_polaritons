using KernelAbstractions

function dispersion(ks, param)
    -im * param.γ / 2 + param.ħ * sum(abs2, ks) / 2param.m - param.δ₀
end

function potential(rs, param)
    param.V_def * exp(-(rs[1] - x_horizon)^2 / param.w_def^2) -
    im * param.V_damp * exp(-(rs[1] - param.L / 2)^2 / param.w_damp^2)
end

heaviside(x) = x > 0

function pump(x, param, t)
    (
        (param.F_p_max - param.F_p_support_u) * sech((x[1] + param.L/2) / param.σ_sech) +
        param.F_p_support_u .* heaviside(param.divide - x[1])
    ) .* cis(param.k_up * x[1]) +
    param.F_p_support_d * heaviside(x[1] - param.divide) *
    cis(param.k_down * x[1])
end

nonlinearity(ψ, param) = param.g * (abs2(first(ψ)) - 1 / param.δL)

noise_func(ψ, param) = √(param.γ / 2 / param.δL)

choose(x1, x2, m) = isone(m) ? x1 : x2

function calculate_momentum_commutators(kernel1, kernel2, L)
    @kernel function kernel!(dest, kernel1, kernel2)
        a, b, m, n = @index(Global, NTuple)
        idx1 = choose(a, b, m)
        idx2 = choose(a, b, n)
        field1 = choose(kernel1, kernel2, m)
        field2 = choose(kernel1, kernel2, n)

        x = zero(eltype(dest))
        for r ∈ axes(kernel1, 2)
            x += field1[idx1, r] * conj(field2[idx2, r])
        end
        dest[a, b, m, n] = x
    end

    dest = similar(kernel1, size(kernel1, 1), size(kernel1, 1), 2, 2)
    backend = get_backend(dest)
    kernel!(backend)(dest, kernel1, kernel2, ndrange=size(dest))

    dest * L / length(kernel1)
end

function calculate_position_commutators(one_point, δL)
    commutators_r = similar(one_point)
    commutators_r[:, :, 1, 1] .= 1 / δL
    commutators_r[:, :, 2, 2] .= 1 / δL
    commutators_r[:, :, 1, 2] .= one(two_point_r) ./ δL
    commutators_r[:, :, 2, 1] .= one(two_point_r) ./ δL

    commutators_r
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