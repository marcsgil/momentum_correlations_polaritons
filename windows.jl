include("io.jl")
include("plot_funcs.jl")

function hamming(N, ::Type{T}) where {T}
    α = 25 / 46
    β = 1 - α
    [T(α - β * cospi(2 * n / (N - 1))) for n ∈ 0:N-1]
end

function hann(N, ::Type{T}) where {T}
    [T(sinpi(n / (N - 1))^2) for n ∈ 0:N-1]
end

saving_dir = "/Users/marcsgil/LEON/MomentumCorrelations/SupportDownstreamRepulsive"

steady_state, param = jldopen(joinpath(saving_dir, "steady_state.jld2")) do file
    file["steady_state"], file["param"]
end
xs = StepRangeLen(0, param.dx, param.N) .- param.x_def
##
window1 = Window(0, 150, xs, hann)
window2 = Window(-150, 0, xs, hann)
save_window_pair(saving_dir, Pair(window1, window2))
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
    window1 = Window(-100, 700, xs, func)
    window2 = Window(-700, 100, xs, func)
    save_window_pair(saving_dir, Pair(window1, window2))
end
##
for func ∈ (hamming, hann)
    window1 = Window(-800, 800, xs, func)
    window2 = Window(-800, 800, xs, func)
    save_window_pair(saving_dir, Pair(window1, window2))
end
##
plot_all_windows(saving_dir, savefig=true)
##