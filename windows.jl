include("io.jl")

function hamming(N, ::Type{T}) where {T}
    [T(0.54 - 0.46 * cospi(2 * n / N)) for n ∈ 0:N-1]
end
##
saving_dir = "/Volumes/partages/EQ15B/LEON-15B/Users/Marcos/MomentumCorrelations/SupportDownstreamRepulsive"

param = jldopen(joinpath(saving_dir, "steady_state.jld2")) do file
    file["param"]
end
rs = StepRangeLen(0, param.δL, param.N) .- param.x_def

window1 = Window(-10, 790, rs, hamming)
window2 = Window(-790, 10, rs, hamming)
save_windows(saving_dir, Pair(window1, window2))