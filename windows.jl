include("io.jl")
include("plot_funcs.jl")

function hamming(N, ::Type{T}) where {T}
    [T(0.54 - 0.46 * cospi(2 * n / (N - 1))) for n ∈ 0:N-1]
end

function hann(N, ::Type{T}) where {T}
    [T(sinpi(n / (N - 1))^2) for n ∈ 0:N-1]
end

saving_dir = "/Volumes/partages/EQ15B/LEON-15B/Users/Marcos/MomentumCorrelations/SupportDownstreamRepulsive"

param = jldopen(joinpath(saving_dir, "steady_state.jld2")) do file
    file["param"]
end
xs = StepRangeLen(0, param.dx, param.N) .- param.x_def
##
for func ∈ (hamming, hann)
    window1 = Window(-10, 790, xs, func)
    window2 = Window(-790, 10, xs, func)
    save_window_pair(saving_dir, Pair(window1, window2))
end
##
for func ∈ (hamming, hann)
    window1 = Window(0, 800, xs, func)
    window2 = Window(-800, 0, xs, func)
    save_window_pair(saving_dir, Pair(window1, window2))
end
##
for func ∈ (hamming, hann)
    window1 = Window(-50, 750, xs, func)
    window2 = Window(-750, 50, xs, func)
    save_window_pair(saving_dir, Pair(window1, window2))
end
##
for func ∈ (hamming, hann)
    window1 = Window(-800, 800, xs, func)
    window2 = Window(-800, 800, xs, func)
    save_window_pair(saving_dir, Pair(window1, window2))
end

#= gauss = @. exp(-(xs / 300)^2)

window1 = Window(gauss, 1)
window2 = Window(gauss, 1) =#
##
plot_all_windows(saving_dir, savefig=true)