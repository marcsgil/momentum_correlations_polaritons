using HDF5

# Parameters

function write_parameters!(parent, param)
    for (key, value) in zip(keys(param), values(param))
        attrs(parent)[string(key)] = value
    end
end

function read_parameters(parent)
    ks = Symbol.(keys(attrs(parent)))
    vals = values(attrs(parent))
    (; (ks .=> vals)...)
end

function get_correlation_buffers(prototype1::Tuple{1}, prototype2::Tuple{1})
    two_point = zero(first(prototype1)) * zero(first(prototype2))'
    one_point = stack(two_point for a ∈ 1:2, b ∈ 1:2)
    one_point, two_point
end

function save_mean_field(steady_state, saving_path, param, group_name, tspan)
    first_order_r, second_order_r = get_correlation_buffers(steady_state, steady_state)
    h5open(saving_path, "cw") do file
        group = create_group(file, group_name)
        write_parameters!(group, param)
        group["steady_state"] = steady_state
        group["t_steady_state"] = tspan[end]
        group["first_order_r"] = first_order_r
        group["second_order_r"] = second_order_r
        group["n_ave"] = [0]
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

struct WindowedFTBuffers{T1,T2,T3,T4,T5}
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
end

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