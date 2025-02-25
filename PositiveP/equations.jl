using StaticArrays

function dispersion(ks, param)
    val = -im * param.γ / 2 + param.ħ * sum(abs2, ks) / 2param.m - param.δ₀
    SVector(val, -conj(val))
end

function potential(rs, param)
    val = param.V_def * exp(-sum(abs2, rs) / param.w_def^2) +
          param.V_damp * damping_potential(rs, -param.L / 2, param.L / 2, param.w_damp)
    SVector(val, -conj(val))
end

function A(t, Amax, t_cycle, t_freeze)
    _t = ifelse(t > t_freeze, t_freeze, t)
    val = Amax * _t * (t_cycle - _t) * 4 / t_cycle^2
    val < 0 ? zero(val) : val
end

function pump(x, param, t)
    a = A(t, param.Amax, param.t_cycle, param.t_freeze)
    if x[1] ≤ -param.L * 0.9 / 2 || x[1] ≥ -10
        a *= 0
    elseif -param.L * 0.9 / 2 < x[1] ≤ -param.L * 0.85 / 2
        a *= 6
    end
    val = a * cis(mapreduce(*, +, param.k_up, x))
    SVector(val, conj(val))
end

function nonlinearity(ψ, param)
    val = param.g * prod(ψ)
    SVector(val, -val)
end

function noise_func(ψ, param)
    val = √(im * param.g)
    SVector(val * ψ[1], conj(val) * ψ[2])
end

function calculate_g2(n, two_point)
    two_point ./ (n .* n')
end