using JLD2

function save_steady_state(saving_dir, steady_state, param, tspan)
    path = joinpath(saving_dir, "steady_state.jld2")

    jldopen(path, "a+") do file
        file["steady_state"] = steady_state
        file["param"] = param
        file["t_steady_state"] = tspan[end]
    end
end

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

function save_windows(saving_dir, pair)
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