using KernelAbstractions

function dispersion(ks, param)
    -im * param.γ / 2 + param.ħ * sum(abs2, ks) / 2param.m - param.δ₀
end

gaussian(x, center, width) = exp(-((x - center) / width)^2)

function potential(rs, param)
    param.V_def * gaussian(rs[1], param.x_def, param.w_def) -
    im * param.V_damp * (gaussian(rs[1], param.L / 2, param.w_damp) +
                         gaussian(rs[1], -param.L / 2, param.w_damp))
end

function half_pump(x, Fmax, Fmin, k, w, L)
    ((Fmax - Fmin) * sech((x + L / 2) / w) + Fmin) * cis(k * x)
end

time_dependence(t, param) = (5 * exp(-t / 100) + 1)

function pump(x, param, t)
    if x[1] < param.divide
        half_pump(x[1], param.F_max, param.F_up, param.k_up, param.w_pump, param.L) * time_dependence(t, param)
    else
        half_pump(-x[1], param.F_max, param.F_down, -param.k_down, param.w_pump, param.L) * time_dependence(t, param)
    end
end

nonlinearity(ψ, param) = param.g * (abs2(first(ψ)) - 1 / param.δL)

noise_func(ψ, param) = √(param.γ / 2 / param.δL)

choose(x1, x2, m) = isone(m) ? x1 : x2

function build_δ(N1, N2, first_idx1, first_idx2)
    map(Iterators.product(1:N1, 1:N2)) do (m, n)
        iszero(m - n + first_idx1 - first_idx2)
    end
end

build_δ(10, 10, 2, 1)

function calculate_momentum_commutators(window1, window2, L1, L2, first_idx1, first_idx2)
    @kernel function kernel!(dest, _window1, _window2, _rs1, _rs2, _ks1, _ks2, _first_idx1, _first_idx2)
        a, b, m, n = @index(Global, NTuple)
        idx1 = choose(a, b, m)
        idx2 = choose(a, b, n)
        window1 = choose(_window1, _window2, m)
        window2 = choose(_window2, _window2, n)
        rs1 = choose(_rs1, _rs2, m)
        rs2 = choose(_rs1, _rs2, n)
        ks1 = choose(_ks1, _ks2, m)
        ks2 = choose(_ks1, _ks2, n)
        first_idx1 = choose(_first_idx1, _first_idx2, m)
        first_idx2 = choose(_first_idx1, _first_idx2, n)

        x = zero(eltype(dest))
        for r ∈ eachindex(rs1, window1)
            s = r + first_idx1 - first_idx2
            if s ∈ eachindex(rs2, window2)
                x += window1[r] * conj(window2[s]) * cis(ks2[idx2] * rs2[s] - ks1[idx1] * rs1[r])
            end
        end
        dest[a, b, m, n] = x
    end

    N1 = length(window1)
    N2 = length(window2)

    rs1 = (0:N1-1) * L1 / N1
    rs2 = (0:N2-1) * L2 / N2
    ks1 = (0:N1-1) * 2π / L1
    ks2 = (0:N2-1) * 2π / L2

    dest = similar(window1, size(window1, 1), size(window2, 1), 2, 2)
    backend = get_backend(dest)
    kernel!(backend)(dest, window1, window2, rs1, rs2, ks1, ks2, first_idx1, first_idx2, ndrange=size(dest))

    dest / L1 / length(window2)
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