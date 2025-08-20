using CairoMakie, JLD2, FFTW
include("io.jl")

function hann(N, ::Type{T}) where {T}
    [T(sinpi(n / N)^2) for n ∈ 0:N-1]
end

function windowed_ft!(dest, src, window_func, first_idx, plan)
    N = length(window_func)
    dest .= view(src, first_idx:first_idx+N-1, :) .* window_func
    plan * dest
end

saving_dir = "data"

steady_state, param = jldopen(joinpath(saving_dir, "steady_state.jld2")) do file
    file["steady_state"], file["param"]
end

window_pairs = read_window_pairs(saving_dir)
##
xs = StepRangeLen(0, param.dx, param.N) .- param.x_def


#window = window_pairs[1].first


#tick_vals = [-1, 0, param.k_up, param.k_down, 1]
#tick_names = ["-1", "0", L"k_{\text{up}}", L"k_{\text{down}}", "1"]

with_theme(theme_latexfonts()) do
    fig = Figure(; size=(1600, 600), fontsize=24)

    for (n, (width, text)) in enumerate(zip(50:50:250, ("(a)", "(b)", "(c)", "(d)", "(e)")))
        center = 0
        #width = 200
        window = Window(center - width / 2, center + width / 2, xs, hann)

        dest = similar(complex(window.window))
        plan = plan_fft!(dest)
        ks = fftshift(fftfreq(length(window.window), 2π / param.dx))
        N2 = param.dx^2 * length(window.window) / 2π / sum(abs2, window.window)

        windowed_ft!(dest, steady_state[1], window.window, window.first_idx, plan)

        energy_momentum = param.ħ * param.g * abs2.(fftshift(dest)) * N2
        energy_position = param.ħ * param.g * abs2.(steady_state[1])


        ax = Axis(fig[1, n], xlabel=L"k \ (\mu \text{m}^{-1})", yscale=log10, ylabel=L"\hbar g|\Phi(k)|^2 \ (\text{meV})", yminorticks=IntervalsBetween(10), yminorticksvisible=true)

        lines!(ax, ks, energy_momentum, linewidth=3)
        xlims!(ax, -0.85, 0.85)
        ylims!(; low=1e-2, high=1e4)
        vlines!(ax, param.k_up, linewidth=2, linestyle=:dash, color=:black)
        vlines!(ax, param.k_down, linewidth=2, linestyle=:dash, color=:black)

        ax2 = Axis(fig[2, n], xlabel=L"x - x_H \ (\mu \text{m})", ylabel=L"\hbar g|\Phi(x)|^2 \ (\text{meV})")
        xlims!(ax2, -130, 130)
        lines!(ax2, xs, energy_position, linewidth=3)
        lines!(ax2, xs[window.first_idx:window.first_idx+length(window.window)-1], window.window * maximum(energy_position), linewidth=3, color=:red, linestyle=:dash, label="Window")

        if n > 1
            hideydecorations!(ax, grid=false)
            hideydecorations!(ax2, grid=false)
        end

        text!(ax, 0.05, 0.85; text, space=:relative, fontsize=24)

        #save("plots/momentum_density_width.pdf", fig)
        fig
    end

    fig
end
