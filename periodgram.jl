using DSP, JLD2, CairoMakie, FFTW

saving_dir = "/Users/marcsgil/LEON/MomentumCorrelations/SupportDownstreamRepulsive1"

steady_state, param, t_steady_state = jldopen(joinpath(saving_dir, "steady_state.jld2")) do file
    file["steady_state"],
    file["param"],
    file["t_steady_state"]
end
##
split = arraysplit(steady_state[1], 200, 190)

p = periodogram(split[54], window = hann)

with_theme(theme_latexfonts()) do
    fig = Figure(; size=(730, 600), fontsize=20)
    ax = Axis(fig[1, 1], xlabel=L"k", ylabel=L"g(k)", title="Periodogram", yscale = log10)
    lines!(ax, fftshift(p.freq), fftshift(p.power))
    ylims!(ax, (1e-2, 1e8))
    fig
end
##
s = spectrogram(steady_state[1], window = hann)

s.time

heatmap(fftshift(s.freq), fftshift(s.time), fftshift(log.(s.power)))