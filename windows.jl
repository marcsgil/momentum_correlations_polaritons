include("io.jl")
include("plot_funcs.jl")

function hamming(N, ::Type{T}) where {T}
    α = 25 / 46
    β = 1 - α
    [T(α - β * cospi(2 * n / N)) for n ∈ 0:N-1]
end

function hann(N, ::Type{T}) where {T}
    [T(sinpi(n / N)^2) for n ∈ 0:N-1]
end

saving_dir = "/home/marcsgil/Code/LEON/MomentumCorrelations/full_sim2"

steady_state, param = jldopen(joinpath(saving_dir, "steady_state.jld2")) do file
    file["steady_state"], file["param"]
end
xs = StepRangeLen(0, param.dx, param.N) .- param.x_def
##
window_length = 500

for start ∈ (-250, )
    xmin = start
    xmax = start + window_length
    window1 = Window(xmin, xmax, xs, hann)
    window2 = Window(-xmax, -xmin, xs, hann)
    save_window_pair(saving_dir, Pair(window1, window2))
end
##
plot_all_windows(saving_dir, savefig=true, xlims=(-300, 300))