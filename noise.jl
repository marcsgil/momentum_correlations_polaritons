using GeneralizedGrossPitaevskii, CUDA, Random, Dates
include("io.jl")
include("equations.jl")
include("correlation_kernels.jl")

saving_dir = "data/long_lifetime"
#saving_dir = "/home/marcsgil/Code/LEON/MomentumCorrelations/full_sim/"
batchsize = 10^4
nbatches = 1
t_sim = 50.0f0
show_progress = isinteractive()
max_datetime = typemax(DateTime)#DateTime(2025, 8, 20, 9, 0)
array_type = CuArray


update_correlations!(saving_dir, batchsize, nbatches, t_sim;
    dispersion, potential, nonlinearity, pump, position_noise_func,
    show_progress, max_datetime, array_type)
