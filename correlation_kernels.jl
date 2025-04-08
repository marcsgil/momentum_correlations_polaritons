using KernelAbstractions, FFTW, Logging, Dates, LinearAlgebra, ProgressMeter, CUDA

choose(x1, x2, m) = isone(m) ? x1 : x2

function kahan_step(x, c, next)
    # Apply Kahan summation algorithm
    y = next - c    # Corrected value (value to be added minus compensation)
    t = x + y          # Raw sum (might lose low-order bits)
    c = (t - x) - y    # Calculate new compensation term
    x = t
    x, c
end

function merge_averages(μ1, n1, μ2, n2)
    μ1 / (1 + n2 / n1) + μ2 / (1 + n1 / n2)
end

@kernel function mean_kernel!(dest, f, n_ave, fields::NTuple{N}) where {N}
    J = @index(Global, NTuple)
    x = zero(eltype(dest))
    c = zero(eltype(dest))

    for m ∈ axes(first(fields), 2)
        next = f(ntuple(n -> fields[n][J[n], m], N))
        x, c = kahan_step(x, c, next)
    end

    dest[J...] = merge_averages(dest[J...], n_ave, x, size(first(fields), 2))
end


function first_order_correlations!(dest, sol1::NTuple{1}, sol2::NTuple{1}, n_ave)
    @kernel function kernel!(dest, sol1, sol2, n_ave)
        a, b, m, n = @index(Global, NTuple)
        idx1 = choose(a, b, m)
        idx2 = choose(a, b, n)
        field1 = choose(sol1, sol2, m)
        field2 = choose(sol1, sol2, n)

        x = zero(eltype(dest))
        c = zero(eltype(dest))  # Compensation term for Kahan summation
        for r ∈ axes(sol1, 2)
            next = field1[idx1, r] * conj(field2[idx2, r])
            x, c = kahan_step(x, c, next)  # Apply Kahan summation
        end
        dest[a, b, m, n] = merge_averages(dest[a, b, m, n], n_ave, x, size(sol1, 2))
    end

    backend = get_backend(dest)
    kernel!(backend)(dest, sol1[1], sol2[1], n_ave, ndrange=size(dest))
end

function second_order_correlations!(dest, sol1::NTuple{1}, sol2::NTuple{1}, n_ave)
    mean_kernel!(get_backend(dest))(dest, fields -> abs2(fields[1]) * abs2(fields[2]), n_ave, (sol1[1], sol2[1]), ndrange=size(dest))
end

function windowed_ft!(dest, src, window_func, first_idx, plan)
    N = length(window_func)
    dest .= view(src, first_idx:first_idx+N-1, :) .* window_func
    plan * dest
end

function init_correlations(saving_dir, steady_state, dx, t_sim)
    jldopen(joinpath(saving_dir, "averages.jld2"), "a+") do file
        file["position_correlations"] = PositionCorrelations(steady_state, dx)
        file["n_ave"] = 0
        file["t_sim"] = t_sim
    end
end

function create_new_then_rename(file_path, new_content)
    parent = dirname(file_path)
    file_name = basename(file_path)
    tmp = tempname(parent)

    new_file = jldopen(tmp, "a+")
    jldopen(file_path) do old_file
        for key ∈ keys(old_file)
            if haskey(new_content, key)
                new_file[key] = new_content[key]
            else
                new_file[key] = old_file[key]
            end
        end
    end
    close(new_file)

    mv(file_path, joinpath(parent, "previous_" * file_name))
    mv(tmp, file_path)

    nothing
end

function update_correlations!(saving_dir, batchsize, nbatches, t_sim;
    noise_eltype=(),
    rng=nothing,
    max_datetime=typemax(DateTime),
    show_progress=true,
    log_path="log.txt",
    kwargs...)

    steady_state, param, t_steady_state = jldopen(joinpath(saving_dir, "steady_state.jld2")) do file
        file["steady_state"], file["param"], file["t_steady_state"]
    end

    saving_path = joinpath(saving_dir, "averages.jld2")
    @assert !isfile(joinpath(saving_dir, "previous_averages.jld2")) "Previous averages file already exists. Please remove it before running the simulation."

    if !isfile(saving_path)
        init_correlations(saving_dir, steady_state, param.dx, t_sim)
    end

    position_correlations, n_ave, t_sim = jldopen(saving_path) do file
        file["position_correlations"], file["n_ave"], file["t_sim"]
    end

    u0 = map(x -> stack(x for _ ∈ 1:batchsize), steady_state)
    noise_prototype = map(x -> similar(x, noise_eltype...), steady_state)
    @show eltype(first(noise_prototype))

    prob = GrossPitaevskiiProblem(u0, (param.L,); noise_prototype, param, kwargs...)
    solver = StrangSplitting()

    io = open(log_path, "w+")
    logger = SimpleLogger(io)

    tspan = (t_steady_state, t_steady_state + t_sim)
    steps_per_save = GeneralizedGrossPitaevskii.resolve_fixed_timestepping(param.dt, tspan, 1)[2]
    if show_progress
        progress = Progress(steps_per_save * nbatches)
    else
        progress = nothing
    end

    for batch ∈ 1:nbatches
        now() > max_datetime && break
        with_logger(logger) do
            @info "Batch $batch"
        end
        flush(io)

        sol = map(GeneralizedGrossPitaevskii.solve(prob, solver, tspan; nsaves=1, dt=param.dt, save_start=false, show_progress, progress, rng)[2]) do x
            dropdims(x, dims=3)
        end

        first_order_correlations!(position_correlations.first_order, sol, sol, n_ave)
        second_order_correlations!(position_correlations.second_order, sol, sol, n_ave)

        n_ave += batchsize
    end

    new_content = Dict(
        "position_correlations" => position_correlations,
        "n_ave" => n_ave,
    )

    create_new_then_rename(saving_path, new_content)

    close(io)
    if !isnothing(progress)
        finish!(progress)
    end

    nothing
end