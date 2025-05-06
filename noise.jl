using GeneralizedGrossPitaevskii, CUDA, Random, Dates
include("io.jl")
include("equations.jl")
include("correlation_kernels.jl")

#saving_dir = "/home/stagios/Marcos/LEON_Marcos/MomentumCorrelations/150_100um_window"
saving_dir = "/home/marcsgil/Code/LEON/MomentumCorrelations/150_100um_window"
batchsize = 10^4
nbatches = 10^6
t_sim = 50.0f0
show_progress = isinteractive()
max_datetime = DateTime(2025, 5, 7, 9, 0)
array_type = CuArray


update_correlations!(saving_dir, batchsize, nbatches, t_sim;
    dispersion, potential, nonlinearity, pump, position_noise_func,
    show_progress, max_datetime, array_type)
