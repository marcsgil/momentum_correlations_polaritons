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

function update_correlations!(first_order_r, second_order_r, first_order_k, second_order_k, n_ave, steady_state, window1, window2, first_idx1, first_idx2,
    lengths, batchsize, nbatches, tspan, dt;
    show_progress=true, noise_eltype=eltype(first(steady_state)), log_path="log.txt", max_datetime=typemax(DateTime),
    rng=nothing, kwargs...)
    u0 = map(steady_state) do x
        stack(x for _ ∈ 1:batchsize)
    end

    noise_prototype = similar.(u0, noise_eltype)

    prob = GrossPitaevskiiProblem(u0, lengths; noise_prototype, param, kwargs...)
    solver = StrangSplitting()

    ft_sol1 = map(steady_state) do x
        stack(window1 for _ ∈ 1:batchsize)
    end
    ft_sol2 = map(steady_state) do x
        stack(window2 for _ ∈ 1:batchsize)
    end

    plan1 = plan_fft!(ft_sol1[1], 1)
    plan2 = plan_fft!(ft_sol2[1], 1)

    io = open(log_path, "w+")
    logger = SimpleLogger(io)

    steps_per_save = GeneralizedGrossPitaevskii.resolve_fixed_timestepping(dt, tspan, 1)[2]
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

        sol = map(GeneralizedGrossPitaevskii.solve(prob, solver, tspan; nsaves=1, dt, save_start=false, show_progress, progress, rng)[2]) do x
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