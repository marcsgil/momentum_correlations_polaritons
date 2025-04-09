using KernelAbstractions, FFTW, Logging, Dates, LinearAlgebra, ProgressMeter

choose(x1, x2, m) = isone(m) ? x1 : x2

function kahan_step(x, c, next)
    # Apply Kahan summation algorithm
    y = next - c    # Corrected value (value to be added minus compensation)
    t = x + y          # Raw sum (might lose low-order bits)
    c = (t - x) - y    # Calculate new compensation term
    x = t
    x, c
end

@kernel function mean_prod_kernel!(dest, src1, src2, f1, f2, n_ave)
    j, k = @index(Global, NTuple)
    x = zero(eltype(dest))
    c = zero(eltype(dest))  # Compensation term for Kahan summation

    for m ∈ axes(src1, 2)
        # Calculate product for this element
        next = f1(src1[j, m]) * f2(src2[k, m])
        x, c = kahan_step(x, c, next)  # Apply Kahan summation
    end

    dest[j, k] = merge_averages(dest[j, k], n_ave, x, size(src1, 2))
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
    mean_prod_kernel!(get_backend(dest))(dest, sol1[1], sol2[1], abs2, abs2, n_ave, ndrange=size(dest))
end

merge_averages(μ, n, new_sum, new_n) = μ / (1 + new_n / n) + new_sum / (n + new_n)

function windowed_ft!(dest, src, window_func, first_idx, plan)
    N = length(window_func)
    dest .= view(src, first_idx:first_idx+N-1, :) .* window_func
    plan * dest
end

function get_correlation_buffers(prototype1::NTuple{1}, prototype2::NTuple{1})
    second_order = zero(complex(first(prototype1))) * zero(complex(first(prototype2)))'
    first_order = stack(second_order for a ∈ 1:2, b ∈ 1:2)
    first_order, second_order
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

function update_correlations!(first_order_r, second_order_r, first_order_k, second_order_k, n_ave, steady_state, param, window_pairs, batchsize, nbatches, tspan;
    show_progress=true, noise_eltype=eltype(first(steady_state)), log_path="log.txt", max_datetime=typemax(DateTime),
    rng=nothing, kwargs...)
    u0 = map(steady_state) do x
        stack(x for _ ∈ 1:batchsize)
    end

    noise_prototype = similar.(u0, noise_eltype)

    prob = GrossPitaevskiiProblem(u0, (param.L,); noise_prototype, param, kwargs...)
    solver = StrangSplitting()

    window1 = window_pairs[1].first.window
    window2 = window_pairs[1].second.window
    first_idx1 = window_pairs[1].first.first_idx
    first_idx2 = window_pairs[1].second.first_idx

    ft_sol1 = map(steady_state) do x
        stack(complex(window1) for _ ∈ 1:batchsize)
    end
    ft_sol2 = map(steady_state) do x
        stack(complex(window2) for _ ∈ 1:batchsize)
    end

    plan1 = plan_fft!(ft_sol1[1], 1)
    plan2 = plan_fft!(ft_sol2[1], 1)

    io = open(log_path, "w+")
    logger = SimpleLogger(io)

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

        sol = map(GeneralizedGrossPitaevskii.solve(prob, solver, tspan; nsaves=1, param.dt, save_start=false, show_progress, progress, rng)[2]) do x
            dropdims(x, dims=3)
        end

        first_order_correlations!(first_order_r, sol, sol, n_ave)
        second_order_correlations!(second_order_r, sol, sol, n_ave)

        for (dest1, dest2, src) ∈ zip(ft_sol1, ft_sol2, sol)
            windowed_ft!(dest1, src, window1, first_idx1, plan1)
            windowed_ft!(dest2, src, window2, first_idx2, plan2)
        end

        first_order_correlations!(first_order_k, ft_sol1, ft_sol2, n_ave)
        second_order_correlations!(second_order_k, ft_sol1, ft_sol2, n_ave)

        n_ave += batchsize
    end

    close(io)
    if !isnothing(progress)
        finish!(progress)
    end

    first_order_r, second_order_r, first_order_k, second_order_k, n_ave
end

function init_correlations(saving_dir, steady_state, t_sim)
    first_order_r, second_order_r = get_correlation_buffers(steady_state, steady_state)

    jldopen(joinpath(saving_dir, "averages.jld2"), "a+") do file
        file["first_order_r"] = Array(first_order_r)
        file["second_order_r"] = Array(second_order_r)
        file["n_ave"] = 0
        file["t_sim"] = t_sim

        jldopen(joinpath(saving_dir, "windows.jld2"), "a+") do window_file
            for n ∈ eachindex(keys(window_file))
                pair = window_file["window_pair_$n"]
                first_order, second_order = get_correlation_buffers((pair.first.window,), (pair.second.window,))
                file["first_order_k_$n"] = Array(first_order)
                file["second_order_k_$n"] = Array(second_order)
            end
        end
    end
end

function update_correlations!(saving_dir, batchsize, nbatches, t_sim; array_type::Type{T}=Array, kwargs...) where {T}
    @assert !isfile(joinpath(saving_dir, "previous_averages.jld2")) "Previous averages file already exists. Please remove it before running the simulation."

    steady_state, param, t_steady_state = jldopen(joinpath(saving_dir, "steady_state.jld2")) do file
        file["steady_state"] .|> T,
        file["param"],
        file["t_steady_state"]
    end

    window_pairs = jldopen(joinpath(saving_dir, "windows.jld2")) do file
        [read_window_pair(file, key, T) for key ∈ keys(file)]
    end

    if !isfile(joinpath(saving_dir, "averages.jld2"))
        init_correlations(saving_dir, steady_state, t_sim)
    end

    first_order_r, second_order_r, first_order_k, second_order_k, n_ave = jldopen(joinpath(saving_dir, "averages.jld2")) do file
        file["first_order_r"] |> T,
        file["second_order_r"] |> T,
        file["first_order_k_1"] |> T,
        file["second_order_k_1"] |> T,
        file["n_ave"]
    end

    tspan = (t_steady_state, t_steady_state + t_sim)

    first_order_r, second_order_r, first_order_k, second_order_k, n_ave = update_correlations!(
        first_order_r, second_order_r, first_order_k, second_order_k, n_ave, steady_state, param, window_pairs, batchsize, nbatches, tspan; param, kwargs...)

    new_content = Dict(
        "first_order_r" => first_order_r |> Array,
        "second_order_r" => second_order_r |> Array,
        "first_order_k_1" => first_order_k |> Array,
        "second_order_k_1" => second_order_k |> Array,
        "n_ave" => n_ave
    )

    create_new_then_rename(joinpath(saving_dir, "averages.jld2"), new_content)
end