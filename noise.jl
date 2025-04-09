using GeneralizedGrossPitaevskii, CUDA, Random, Dates
include("io.jl")
include("equations.jl")
include("correlation_kernels.jl")

saving_dir = "/home/stagios/Marcos/LEON_Marcos/Users/Marcos/MomentumCorrelations/SupportDownstreamRepulsive"
batchsize = 100000
nbatches = 10
t_sim = 50.0f0
show_progress = true
max_datetime = DateTime(2025, 4, 9, 17, 30)
array_type = CuArray


update_correlations!(saving_dir, batchsize, nbatches, t_sim;
    dispersion, potential, nonlinearity, pump, noise_func,
    show_progress, max_datetime, array_type)