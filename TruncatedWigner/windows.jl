include("../io.jl")

function hamming(n, N, ::Type{T}) where {T}
    T(0.54 - 0.46 * cospi(2 * n / N))
end

saving_path = "/home/stagios/Marcos/LEON_Marcos/Users/Marcos/MomentumCorrelations/TruncatedWigner/correlations.h5"
group_name = "test2"

rs, x_def = h5open(saving_path, "r") do file
    group = file[group_name]
    param = read_parameters(group)
    StepRangeLen(0f0, param.δL, param.N), param.x_def
end

L_window = 800f0

windows_up = [Window(x_begin, x_begin + L_window, rs, hamming) for x_begin in (x_def - L_window / 2, x_def - 10, x_def, x_def + 10)]
windows_down = [Window(x_end - L_window, x_end, rs, hamming) for x_end in (x_def + L_window / 2, x_def + 10, x_def, x_def - 10)]

save_windows(saving_path, group_name, windows_down, windows_up)