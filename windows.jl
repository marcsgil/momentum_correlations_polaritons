include("io.jl")

function hamming(N, ::Type{T}) where {T}
    [T(0.54 - 0.46 * cospi(2 * n / N)) for n ∈ 0:N-1]
end

saving_dir = "/Volumes/partages/EQ15B/LEON-15B/Users/Marcos/MomentumCorrelations/SupportDownstreamRepulsive"

param = jldopen(joinpath(saving_dir, "steady_state.jld2")) do file
    file["param"]
end
xs = StepRangeLen(0, param.dx, param.N) .- param.x_def

window1 = Window(-10, 790, xs, hamming)
window2 = Window(-790, 10, xs, hamming)
save_window_pair(saving_dir, Pair(window1, window2))