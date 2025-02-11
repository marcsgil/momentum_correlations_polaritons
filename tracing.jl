using Roots, ForwardDiff

function find_extrema(f, bracket, args...)
    ext = find_zero(x -> ForwardDiff.derivative(k -> f(k, args...), x), bracket, Bisection())
    ext, f(ext, args...)
end