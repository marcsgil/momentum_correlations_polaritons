using JLD2

#Steady state

function save_steady_state(saving_dir, steady_state, param, tspan)
    path = joinpath(saving_dir, "steady_state.jld2")

    jldopen(path, "a+") do file
        file["steady_state"] = steady_state
        file["param"] = param
        file["t_steady_state"] = tspan[end]
    end
end

function read_steady_state(saving_dir, ::Type{T}) where {T}
    path = joinpath(saving_dir, "steady_state.jld2")

    jldopen(path) do file
        file["steady_state"] .|> T,
        file["param"],
        file["t_steady_state"]
    end
end

# Windows

struct Window{T<:AbstractVector}
    window::T
    first_idx::Int
end

position2idx(x, rs) = argmin(idx -> abs(rs[idx] - x), eachindex(rs))

function Window(x_begin, x_end, rs, func)
    first_idx = position2idx(x_begin, rs)
    last_idx = position2idx(x_end, rs)
    N = last_idx - first_idx
    window = func(N, eltype(first(rs)))
    Window(window, first_idx)
end

function save_window_pair(saving_dir, pair)
    path = joinpath(saving_dir, "windows.jld2")

    jldopen(path, "a+") do file
        n = length(keys(file)) + 1
        file["window_pair_$n"] = pair
    end
end

convert_window(window, ::Type{T}) where {T} = Window(T(window.window), window.first_idx)

function read_window_pair(file, key, ::Type{T}) where {T}
    pair = file[key]
    Pair(convert_window(pair.first, T), convert_window(pair.second, T))
end

function read_window_pairs(saving_dir, ::Type{T}) where {T}
    path = joinpath(saving_dir, "windows.jld2")
    jldopen(path) do file
        [read_window_pair(file, key, T) for key ∈ keys(file)]
    end
end

# Averages

function get_correlation_buffers(prototype1::NTuple{1}, prototype2::NTuple{1})
    second_order = zero(complex(first(prototype1))) * zero(complex(first(prototype2)))'
    first_order = stack(second_order for a ∈ 1:2, b ∈ 1:2)
    first_order, second_order
end

function init_correlations(saving_dir, steady_state, t_sim)
    path = joinpath(saving_dir, "averages.jld2")
    isfile(path) && return

    first_order_x, second_order_x = get_correlation_buffers(steady_state, steady_state)

    jldopen(path, "a+") do file
        file["first_order_x"] = Array(first_order_x)
        file["second_order_x"] = Array(second_order_x)
        file["n_ave"] = 0
        file["t_sim"] = t_sim

        window_path = joinpath(saving_dir, "windows.jld2")
        !isfile(window_path) && return

        jldopen(window_path) do window_file
            for n ∈ eachindex(keys(window_file))
                pair = window_file["window_pair_$n"]
                first_order, second_order = get_correlation_buffers((pair.first.window,), (pair.second.window,))
                file["first_order_k_$n"] = Array(first_order)
                file["second_order_k_$n"] = Array(second_order)
            end
        end
    end

    nothing
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