using GeneralizedGrossPitaevskii, CUDA, Random
include("io.jl")
include("equations.jl")
include("correlation_kernels.jl")

saving_dir = "/Volumes/partages/EQ15B/LEON-15B/Users/Marcos/MomentumCorrelations/SupportDownstreamRepulsive"
t_sim = 50.0f0

update_correlations!(saving_dir, 10^2, 10, t_sim;
    dispersion, potential, nonlinearity, pump, noise_func, show_progress=true,
    max_datetime=DateTime(2025, 4, 10, 10, 0));