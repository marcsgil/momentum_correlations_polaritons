using ArgParse, GeneralizedGrossPitaevskii, CUDA, Random, Dates
include("io.jl")
include("equations.jl")
include("correlation_kernels.jl")

#= Example usage:

julia --project noise.jl --saving_dir /home/stagios/Marcos/LEON_Marcos/Users/Marcos/MomentumCorrelations/SupportDownstreamRepulsive 
--array_type CuArray  --batchsize 100000 --nbatches 10 --t_sim 50 --show_progress --max_datetime 2025,04,09,14,30
=#

function parse_commandline()
    s = ArgParseSettings(description="Run noise correlation simulations for polaritions.")

    @add_arg_table s begin
        "--saving_dir", "-d"
        help = "Directory for saving simulation results"
        required = true
        arg_type = String

        "--batchsize", "-b"
        help = "Size of each simulation batch"
        required = true
        arg_type = Int

        "--nbatches", "-n"
        help = "Number of batches to run"
        required = true
        arg_type = Int

        "--t_sim", "-t"
        help = "Simulation time"
        required = true
        arg_type = Float32

        "--show_progress", "-p"
        help = "Show progress bar"
        action = :store_true

        "--max_datetime", "-m"
        help = "Maximum datetime to run (format: YYYY, MM, DD, HH, MM)"
        arg_type = String
        default = ""

        "--array_type", "-a"
        help = "Array type to use (Array or CuArray)"
        arg_type = String
        default = "Array"
    end

    return parse_args(s)
end

function main()
    args = parse_commandline()

    saving_dir = args["saving_dir"]
    batchsize = args["batchsize"]
    nbatches = args["nbatches"]
    t_sim = args["t_sim"]
    show_progress = args["show_progress"]

    # Parse max_datetime if provided
    max_datetime = if args["max_datetime"] == ""
        typemax(DateTime)
    else
        DateTime(Tuple(parse.(Int, split(args["max_datetime"], ',')))...)
    end

    # Parse array_type
    array_type = if args["array_type"] == "CuArray"
        CuArray
    elseif args["array_type"] == "Array"
        Array
    else
        error("Invalid array_type. Must be 'Array' or 'CuArray'")
    end

    # Run the simulation
    update_correlations!(saving_dir, batchsize, nbatches, t_sim;
        dispersion, potential, nonlinearity, pump, noise_func,
        show_progress, max_datetime, array_type)

    println("Simulation completed successfully.")
end

main()