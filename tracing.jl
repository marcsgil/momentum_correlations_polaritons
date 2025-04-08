using Roots, ForwardDiff
include("polariton_funcs.jl")

function find_extrema(f, bracket, args...)
    ext = find_zero(x -> ForwardDiff.derivative(k -> f(k, args...), x), bracket, Bisection())
    ext, f(ext, args...)
end

function correlate(param1, bracket1, param2, bracket2, N, invert_ω::Bool=false)
    ks1 = Vector(LinRange(bracket1..., N))
    ks2 = similar(ks1)

    for n ∈ eachindex(ks2)
        ω1 = dispersion_relation(ks1[n], param1...)
        try 
            ks2[n] = find_zero(k′ -> dispersion_relation(k′, param2...) + (2invert_ω - 1) * ω1, bracket2)
        catch
            ks2[n] = NaN
        end
    end

    ks1, ks2
end