using GeneralizedGrossPitaevskii, CUDA, Random, Dates
include("io.jl")
include("equations.jl")
include("correlation_kernels.jl")

#saving_dir = "data/test"
saving_dir = "/home/stagios/Marcos/LEON_Marcos/MomentumCorrelations/test"
batchsize = 10^4
nbatches = 81
t_sim = 50.0f0
show_progress = isinteractive()
max_datetime = typemax(DateTime)#DateTime(2025, 8, 20, 9, 0)
array_type = CuArray


update_correlations!(saving_dir, batchsize, nbatches;
    dispersion, potential, nonlinearity, pump, position_noise_func,
    show_progress, max_datetime, array_type)
