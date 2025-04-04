using CUDA, Statistics, BenchmarkTools

f(x, y) = x * conj(y)

function run!(dest, sol1, sol2)
    for n ∈ axes(dest, 2)
        slice2 = view(sol2, n, :)
        for m ∈ axes(dest, 1)
            slice1 = view(sol1, m, :)
            @async dest[m, n] = mapreduce(f, +, slice1, slice2) / size(sol1, 2)
        end
    end
end

sol1 = CUDA.randn(ComplexF32, 1024, 10^5)
sol2 = CUDA.randn(ComplexF32, 1024, 10^5)

dest = Array{ComplexF32}(undef, size(sol1, 1), size(sol2, 1))

run!(dest, sol1, sol2)

sum