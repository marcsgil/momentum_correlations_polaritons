using GeneralizedGrossPitaevskii
include("../io.jl")
include("equations.jl")
include("../correlation_kernels.jl")

saving_path = "/home/stagios/Marcos/LEON_Marcos/Users/Marcos/MomentumCorrelations/TruncatedWigner/correlations.h5"
group_name = "test2"

#reset_simulation!(saving_path, group_name)

tspan = (0.0f0, 50.0f0)

max_datetime = DateTime(2025, 4, 7, 11, 0)

update_correlations!(saving_path, group_name, 10^5, 1, tspan; show_progress=true, max_datetime);