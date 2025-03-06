using KernelAbstractions

function dispersion(ks, param)
    -im * param.γ / 2 + param.ħ * sum(abs2, ks) / 2param.m - param.δ₀
end

function potential(rs, param)
    param.V_def * exp(-sum(abs2, rs) / param.w_def^2) +
    param.V_damp * damping_potential(rs, -param.L / 2, param.L / 2, param.w_damp)
end

nonlinearity(ψ, param) = param.g * abs2(ψ)

function A(t, Amax, t_cycle, t_freeze)
    _t = ifelse(t > t_freeze, t_freeze, t)
    val = Amax * _t * (t_cycle - _t) * 4 / t_cycle^2
    val < 0 ? zero(val) : val
end

function pump(x, param, t)
    a = A(t, param.Amax, param.t_cycle, param.t_freeze)

    if abs(x[1]) ≥ param.L * 0.85 / 2
        a *= 0
    elseif -param.L * 0.80 / 2 ≥ x[1] > -param.L * 0.85 / 2
        a *= 6
    end

    if x[1] > param.divide
        a *= param.factor
    end

    k = x[1] < param.divide ? param.k_up : param.k_down

    a * cis(mapreduce(*, +, k, x))
end

noise_func(ψ, param) = √(param.γ / 2 / param.δL)

function calculate_momentum_commutators(windows, L)
    @kernel function kernel!(dest, windows, rs, ks)
        a, b = @index(Global, NTuple)
        for n ∈ axes(dest, 4), m ∈ axes(dest, 3)
            x = zero(eltype(dest))
            k_vals = (ks[a], ks[b])
            Δk = k_vals[n] - k_vals[m]

            for o ∈ axes(windows, 1)
                x += cis(Δk * rs[o]) * windows[o, m] * conj(windows[o, n])
            end
            dest[a, b, m, n] = x
        end
    end

    N = size(windows, 1)
    rs = range(; start=-L / 2, step=L / N, length=N)
    ks = range(; start=-π * N / L, step=2π / L, length=N)
    dest = similar(windows, complex(eltype(windows)), N, N, 2, 2)
    backend = get_backend(dest)
    kernel!(backend)(dest, windows, rs, ks, ndrange=size(dest)[1:2])

    dest * L / N^2
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