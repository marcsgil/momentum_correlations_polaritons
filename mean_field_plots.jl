using JLD2

include("plot_funcs.jl")
include("io.jl")

saving_dir = "/home/marcsgil/Code/LEON/MomentumCorrelations/Brasil"

steady_state, param, = read_steady_state(saving_dir)
##
N = param.N
L = param.L
dx = param.dx
xs = StepRangeLen(0, dx, N) .- param.x_def

plot_bistability(xs, steady_state[1], param, -200, 200; saving_dir, factor_ns_down=1000, factor_ns_up=1.35)
##
ks_up = LinRange(-0.7, 0.7, 512)
ks_down = LinRange(-1.5, 1.5, 512)
plot_dispersion(xs, steady_state[1], param, -200, 200, 0.4, ks_up, ks_down; saving_dir)
##
plot_velocities(xs, steady_state[1], param; xlims=(-150, 150), ylims=(0, 3), saving_dir)