using GeneralizedGrossPitaevskii, CUDA, Random, Dates
include("io.jl")
include("equations_simple.jl")
include("correlation_kernels.jl")

CUDA.device!(0); # Select GPU device
CUDA.device()

#saving_dir = "data/test"
saving_dir = "/home/marcsgil/Data/momentum_correlation_polaritons/simple"
batchsize = 10^3
nbatches = 1
show_progress = isinteractive()
max_datetime = typemax(DateTime)#DateTime(2025, 8, 20, 9, 0)
array_type = CuArray


update_correlations!(saving_dir, batchsize, nbatches;
    dispersion, potential, nonlinearity, pump, position_noise_func,
    show_progress, max_datetime, array_type)
