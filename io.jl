using JLD2

struct PositionCorrelations{T1,T2,T3}
    first_order::T1
    second_order::T2
    commutators::T3
end

function get_correlation_buffers(prototype1::NTuple{1}, prototype2::NTuple{1})
    second_order = zero(first(prototype1)) * zero(first(prototype2))'
    first_order = stack(second_order for a ∈ 1:2, b ∈ 1:2)
    first_order, second_order
end

function calculate_position_commutators(one_point::Array{T,4}, dx) where {T}
    commutators = similar(one_point)
    commutators[:, :, 1, 1] .= 1 / dx
    commutators[:, :, 2, 2] .= 1 / dx
    commutators[:, :, 1, 2] .= one(view(commutators, :, :, 1, 2)) ./ dx
    commutators[:, :, 2, 1] .= commutators[:, :, 1, 2]
    commutators
end

function PositionCorrelations(prototype::NTuple{1}, dx)
    first_order, second_order = get_correlation_buffers(prototype, prototype)
    commutators = calculate_position_commutators(first_order, dx)
    PositionCorrelations(first_order, second_order, commutators)
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
    window = func.(0:N-1, N, eltype(first(rs)))
    Window(window, first_idx)
end

function save_windows(saving_path, group_name, windows_down, windows_up)
    for (n, (win_down, win_up)) ∈ enumerate(zip(windows_down, windows_up))
        h5open(saving_path, "cw") do file
            group = file[group_name]
            group["window_down$n"] = win_down.window
            group["window_up$n"] = win_up.window
            attrs(group["window_down$n"])["first_idx"] = win_down.window
            attrs(group["window_up$n"])["first_idx"] = win_up.window
            one_point, two_point = get_correlation_buffers(complex(win_down.window), complex(win_up.window))
            group["one_point_k$n"] = one_point
            group["two_point_k$n"] = two_point
        end
    end
end

function read_window(object, window_name, finalizer=identity)
    window = object[window_name]
    first_idx = attrs(window)["first_idx"]
    Window(finalizer(read(window)), first_idx)
end

#= struct WindowedFTBuffers{T1,T2,T3,T4,T5}
    window::T1
    buffer_first_order::T2
    buffer_second_order::T3
    ft_buffer::T4
    plan::T5

    function WindowedFTBuffers(window, first_order, second_order, steady_state, batchsize)
        buffer_first_order = similar(first_order)
        buffer_second_order = similar(second_order)
        ft_buffer = map(steady_state) do x
            stack(window for _ ∈ 1:batchsize)
        end
        plan = plan_fft!(ft_buffer[1], 1)
        new(window, buffer_first_order, buffer_second_order, ft_buffer, plan)
    end
end =#

function reset_simulation!(saving_path, group_name)
    h5open(saving_path, "cw") do file
        group = file[group_name]
        group["one_point_r"][:, :, :, :] = 0
        group["two_point_r"][:, :] = 0
        group["one_point_k"][:, :, :, :] = 0
        group["two_point_k"][:, :] = 0
        group["n_ave"][:] = 0
    end
    nothing
end