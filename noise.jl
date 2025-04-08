using GeneralizedGrossPitaevskii, JLD2
include("io.jl")
include("equations.jl")
include("correlation_kernels.jl")

saving_dir = "/Volumes/partages/EQ15B/LEON-15B/Users/Marcos/MomentumCorrelations/SupportDownstreamRepulsive"
max_datetime = DateTime(2025, 4, 8, 17, 0)
t_sim = 50f0

update_correlations!(saving_dir, 1000, 1, t_sim;
    max_datetime,
    show_progress=true,
    log_path="log.txt",
    dispersion, potential, nonlinearity, pump, noise_func)
