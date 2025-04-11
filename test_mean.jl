using KernelAbstractions, Statistics

@kernel function mean_kernel!(dest, src, n)
    j = @index(Global)
    μ = dest[j]

    N = n

    for m ∈ axes(src, 2)
        N += 1
        μ += (abs2(src[j, m]) - μ) / N
    end

    dest[j] = μ
end

src = rand(ComplexF32, 1000, 1000)
dest = zeros(ComplexF32, 1000)

mean_kernel!(get_backend(src))(dest, src, 0; ndrange=size(dest))

dest ≈ mean(abs2, src, dims=2)

